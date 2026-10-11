#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端基线（纯 Python + onnxruntime CPU），与 Rust 版同一套管线逻辑：
  preprocess(wav->mel) -> 文本 token 化 -> Euler ODE(CFG+sway) -> vocos_spec -> ISTFT -> wav

用途：
  1. 测 NFE=32/16/8 与 FP32/FP16/INT8 的耗时与 RTF（本机 2 核，仅作相对参考）；
  2. 给 Rust 版做数值对照（同 seed 时噪声 RNG 与 Rust 不一致，只比统计量/相关性）。

示例：
  python baseline.py --dit ../models/dit_step_int8.onnx --nfe 8 \
      --ref-audio ../export/DiaMoE-TTS/prompts/cantonese_female_prompt.wav \
      --ref-text "..." --text "..." --out /tmp/base.wav
"""

import argparse
import os
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf

SR = 24000
HOP = 256
N_FFT = 1024
TARGET_RMS = 0.1


# ---------------- 文本 token 化（与 Rust 版一致：词表最长匹配） ----------------

class Tokenizer:
    def __init__(self, vocab_file):
        with open(vocab_file, encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
        self.map = {tok: i for i, tok in enumerate(lines)}
        self.max_len = max(len(t) for t in lines if t)

    def encode(self, s):
        ids, i = [], 0
        while i < len(s):
            hit = None
            for ln in range(min(self.max_len, len(s) - i), 0, -1):
                sub = s[i:i + ln]
                if sub in self.map:
                    hit = sub
                    break
            if hit is None:
                ids.append(0)  # 未知 -> 0（官方 list_str_to_idx 的 get(c, 0)）
                i += 1
            else:
                ids.append(self.map[hit])
                i += len(hit)
        return ids


# ---------------- ISTFT（torch.istft(center=True) 等价实现） ----------------

def istft(spec, n_fft=N_FFT, hop=HOP):
    """spec: complex [513, F] -> wav [ (F-1)*hop ]"""
    window = np.hanning(n_fft + 1)[:-1]  # periodic Hann，与 torch.hann_window 一致
    frames = np.fft.irfft(spec, n=n_fft, axis=0) * window[:, None]
    f_count = spec.shape[1]
    length = (f_count - 1) * hop + n_fft
    y = np.zeros(length)
    env = np.zeros(length)
    w2 = window ** 2
    for i in range(f_count):
        s = i * hop
        y[s:s + n_fft] += frames[:, i]
        env[s:s + n_fft] += w2
    nz = env > 1e-11
    y[nz] /= env[nz]
    pad = n_fft // 2
    return y[pad:-pad] if pad else y


def sway_timesteps(steps, coef):
    t = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)
    if coef is not None:
        t = t + coef * (np.cos(np.pi / 2 * t) - 1 + t)
    return t.astype(np.float32)


class Pipeline:
    def __init__(self, model_dir, dit_name, threads=2):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.pre = ort.InferenceSession(os.path.join(model_dir, "preprocess.onnx"), so,
                                        providers=["CPUExecutionProvider"])
        self.dit = ort.InferenceSession(os.path.join(model_dir, dit_name), so,
                                        providers=["CPUExecutionProvider"])
        voc = "vocos_spec_fp16.onnx" if os.path.exists(os.path.join(model_dir, "vocos_spec_fp16.onnx")) \
            else "vocos_spec.onnx"
        self.voc = ort.InferenceSession(os.path.join(model_dir, voc), so,
                                        providers=["CPUExecutionProvider"])

    def mel_of(self, wav):
        mel = self.pre.run(None, {"wav": wav[None, :].astype(np.float32)})[0]  # [1,100,F]
        return mel[0].T  # [F,100]

    def run(self, ref_wav, ref_ids, gen_ids, ref_words, gen_words,
            nfe=32, cfg=2.0, sway=-1.0, seed=42, speed=1.0, fix_frames=None):
        t0 = time.time()
        rms = float(np.sqrt(np.mean(ref_wav ** 2)) + 1e-9)
        audio = ref_wav * (TARGET_RMS / rms) if rms < TARGET_RMS else ref_wav
        cond_mel = self.mel_of(audio)  # [F,100]
        ref_len = cond_mel.shape[0]

        if fix_frames is not None:
            duration = fix_frames
        else:
            duration = ref_len + int(ref_len / max(1, ref_words) * gen_words / speed)
        duration = max(duration, ref_len + 1)

        full_ids = ref_ids + [0] + gen_ids
        gate_ids = [-1] * (len(ref_ids) + 1) + gen_ids
        neg_ids = [-1] * len(full_ids)

        def pad(ids):
            arr = np.full((1, duration), -1, dtype=np.int64)
            arr[0, :min(len(ids), duration)] = ids[:duration]
            return arr

        text = pad(full_ids)
        text_gate = pad(gate_ids)
        text_neg = pad(neg_ids)
        positions = np.arange(duration, dtype=np.int64)

        cond = np.zeros((1, duration, 100), dtype=np.float32)
        cond[0, :ref_len] = cond_mel
        zeros = np.zeros_like(cond)

        rng = np.random.default_rng(seed)
        x = rng.standard_normal((1, duration, 100), dtype=np.float32)

        ts = sway_timesteps(nfe, sway)
        for i in range(nfe):
            t = np.array([ts[i]], dtype=np.float32)
            feed = {"x": x, "cond": cond, "text": text, "text_gate": text_gate,
                    "time": t, "moe_flag": np.array([1.0], dtype=np.float32),
                    "positions": positions}
            pred = self.dit.run(None, feed)[0]
            if cfg >= 1e-5:
                feed = {"x": x, "cond": zeros, "text": text_neg, "text_gate": text_neg,
                        "time": t, "moe_flag": np.array([0.0], dtype=np.float32),
                        "positions": positions}
                null = self.dit.run(None, feed)[0]
                vel = pred + (pred - null) * cfg
            else:
                vel = pred
            x = x + (ts[i + 1] - ts[i]) * vel

        x[0, :ref_len] = cond_mel  # 官方：条件段回填参考 mel
        gen_mel = x[0, ref_len:].T[None, :, :]  # [1,100,G]
        spec = self.voc.run(None, {"mel": gen_mel.astype(np.float32)})[0][0]  # [1026,G]
        complex_spec = spec[:513] + 1j * spec[513:]
        wav = istft(complex_spec)
        if rms < TARGET_RMS:
            wav = wav * (rms / TARGET_RMS)
        elapsed = time.time() - t0
        return wav.astype(np.float32), elapsed, duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
    ap.add_argument("--dit", default="dit_step.onnx")
    ap.add_argument("--vocab", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "DiaMoE-TTS", "diamoe_tts", "data", "vocab.txt"))
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--sway", type=float, default=-1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    wav, sr = sf.read(args.ref_audio, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:  # 官方管线同样会把参考音频重采样到 24kHz
        import torch
        import torchaudio

        wav = torchaudio.transforms.Resample(sr, SR)(torch.from_numpy(wav)).numpy()

    tok = Tokenizer(args.vocab)
    ref_ids = tok.encode(args.ref_text)
    gen_ids = tok.encode(args.text)

    pipe = Pipeline(os.path.abspath(args.model_dir), args.dit, threads=args.threads)
    out, elapsed, duration = pipe.run(
        wav, ref_ids, gen_ids,
        ref_words=len(args.ref_text.split()), gen_words=len(args.text.split()),
        nfe=args.nfe, cfg=args.cfg, sway=args.sway, seed=args.seed, speed=args.speed,
    )
    audio_sec = len(out) / SR
    total_sec = duration * HOP / SR
    print(f"dit={args.dit} nfe={args.nfe} 总时长={total_sec:.2f}s 生成={audio_sec:.2f}s "
          f"耗时={elapsed:.2f}s RTF(生成)={elapsed / max(audio_sec, 1e-9):.3f}")
    print(f"输出统计: RMS={float(np.sqrt(np.mean(out ** 2))):.5f} peak={float(np.max(np.abs(out))):.4f} "
          f"samples={len(out)}")
    if args.out:
        sf.write(args.out, out, SR)
        print("已写", args.out)


if __name__ == "__main__":
    main()
