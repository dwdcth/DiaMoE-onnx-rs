#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 faster-whisper 转写合成结果，客观校验 TTS 输出内容（渗漏/重复/吞字检测）。

依赖（venv 已装）：faster-whisper、soundfile、torch/torchaudio（重采样）。
模型首次运行会从 HF 下载 whisper small（int8，~500MB）。

用法：
  .venv/bin/python export/check_asr.py a.wav b.wav [--expect "豫章故郡洪都新府星分翼轸地接衡庐"]
"""

import argparse

import numpy as np
import soundfile as sf
import torch
import torchaudio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--expect", default=None, help="期望文本（去标点后比对提示用）")
    ap.add_argument("--model", default="small")
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    rs = torchaudio.transforms.Resample(24000, 16000)
    if args.expect:
        print(f"期望: {args.expect}")
    for path in args.wavs:
        wav, sr = sf.read(path, dtype="float32")
        if sr != 16000:
            wav = rs(torch.from_numpy(np.ascontiguousarray(wav))[None, :]).numpy().squeeze()
        segs, _ = model.transcribe(wav, language="zh", beam_size=5, vad_filter=True)
        txt = "".join(s.text for s in segs)
        for ch in "，。、 ！？,.!?":
            txt = txt.replace(ch, "")
        print(f"{path}: [{txt}]")
        # 能量填充率（语音是否挤在前面/拖尾）
        win = int(16000 * 0.1)
        env = np.array([np.sqrt((wav[i:i + win] ** 2).mean() + 1e-12)
                        for i in range(0, len(wav) - win, win)])
        speech = env > env.max() * 0.05
        last = np.max(np.nonzero(speech)) + 1 if speech.any() else 0
        print(f"  有声至 {last / len(env):.0%}，有效 {len(wav) / 16000 * last / len(env):.2f}s")


if __name__ == "__main__":
    main()
