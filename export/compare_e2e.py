#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端量化误差对比：两个已生成的 wav（同 seed/同条件、不同 DiT 精度）之间
计算波形 cos、RMS 差，以及经 preprocess.onnx 提取 log mel 后的 L1/cos。

用法：python compare_e2e.py <wav_a> <wav_b> [--models ../models]
"""

import argparse
import os

import numpy as np
import onnxruntime as ort
import soundfile as sf


def mel_of(sess, wav):
    return sess.run(None, {"wav": wav[None, :].astype(np.float32)})[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav_a")
    ap.add_argument("wav_b")
    ap.add_argument("--models", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
    args = ap.parse_args()

    a, sr_a = sf.read(args.wav_a, dtype="float32")
    b, sr_b = sf.read(args.wav_b, dtype="float32")
    assert sr_a == sr_b == 24000
    n = min(len(a), len(b))
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)

    wav_cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    rms_a = float(np.sqrt(np.mean(a ** 2)))
    rms_b = float(np.sqrt(np.mean(b ** 2)))

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(os.path.join(os.path.abspath(args.models), "preprocess.onnx"),
                                so, providers=["CPUExecutionProvider"])
    ma = mel_of(sess, a).astype(np.float64)
    mb = mel_of(sess, b).astype(np.float64)
    mel_l1 = float(np.mean(np.abs(ma - mb)))
    fa, fb = ma.ravel(), mb.ravel()
    mel_cos = float(fa @ fb / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-12))

    print(f"波形: cos={wav_cos:.6f} RMS {rms_a:.5f} vs {rms_b:.5f} (差 {abs(rms_a - rms_b):.5f})")
    print(f"log mel: L1={mel_l1:.4f} cos={mel_cos:.6f}")


if __name__ == "__main__":
    main()
