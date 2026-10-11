#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX 图与 PyTorch 参考的数值对齐验证。

- dit_step: 用导出的参考输入跑 ORT 比对；再以不同序列长度、moe_flag=0/1 各跑一次，
  与 PyTorch 模型在相同输入下的输出比对（验证动态轴与门控分支语义）。
- vocos_spec: 用导出参考输入比对。
误差指标：max abs、mean abs、cosine similarity。
"""

import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def metrics(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return dict(max_abs=float(np.max(np.abs(a - b))), mean_abs=float(np.mean(np.abs(a - b))), cos=cos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights", "10ep_mlpEXP_9.pt"))
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    models = os.path.abspath(args.models)

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads

    # ---- vocos_spec 参考对齐 ----
    ref = torch.load(os.path.join(models, "vocos_spec_ref.pt"), weights_only=False)
    sess = ort.InferenceSession(os.path.join(models, "vocos_spec.onnx"), so, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"mel": ref["input"]})[0]
    print("vocos_spec 参考对齐:", metrics(ref["output"], out))

    # ---- dit_step 参考对齐（导出时保存的输入/输出） ----
    ref = torch.load(os.path.join(models, "dit_step_ref.pt"), weights_only=False)
    sess = ort.InferenceSession(os.path.join(models, "dit_step.onnx"), so, providers=["CPUExecutionProvider"])
    names = ["x", "cond", "text", "text_gate", "time", "moe_flag", "positions"]
    feed = {n: v for n, v in zip(names, ref["inputs"])}
    out = sess.run(None, feed)[0]
    print("dit_step 参考对齐 (N=256, moe=1):", metrics(ref["output"], out))

    # ---- preprocess 参考对齐 + 与官方 MelSpec 对比 ----
    ref = torch.load(os.path.join(models, "preprocess_ref.pt"), weights_only=False)
    sess_pre = ort.InferenceSession(os.path.join(models, "preprocess.onnx"), so, providers=["CPUExecutionProvider"])
    out = sess_pre.run(None, {"wav": ref["input"]})[0]
    print("preprocess 参考对齐:", metrics(ref["output"], out))

    # ---- 变长 + 两分支 vs PyTorch ----
    from export_models import build_model, DiTStep, Preprocess

    model = build_model(args.ckpt, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "DiaMoE-TTS", "diamoe_tts", "data", "vocab.txt"))
    step = DiTStep(model.transformer).eval()

    torch.manual_seed(42)
    for n in (180, 320):
        for moe_flag in (1.0, 0.0):
            inputs = (
                torch.randn(1, n, 100),
                torch.randn(1, n, 100),
                torch.randint(0, 2000, (1, n), dtype=torch.int64),
                torch.randint(0, 2000, (1, n), dtype=torch.int64),
                torch.tensor([0.37]),
                torch.tensor([moe_flag]),
                torch.arange(n, dtype=torch.int64),
            )
            with torch.no_grad():
                # DiTStep.forward 返回 (vel, gate_idx) 元组，这里只比对 vel
                pt_out = step(*inputs)[0].numpy()
            feed = {k: v.numpy() for k, v in zip(names, inputs)}
            ort_out = sess.run(None, feed)[0]
            print(f"dit_step 变长对齐 (N={n}, moe={moe_flag}):", metrics(pt_out, ort_out))

    # ---- 自定义 mel 与官方 MelSpec 的等价性 ----
    wav = torch.randn(1, 24000 * 2)
    with torch.no_grad():
        off = model.mel_spec(wav).numpy()
        cus = Preprocess().eval()(wav).numpy()
    print("preprocess vs 官方MelSpec:", metrics(off, cus), "shape", off.shape)

    print("验证完成")


if __name__ == "__main__":
    main()
