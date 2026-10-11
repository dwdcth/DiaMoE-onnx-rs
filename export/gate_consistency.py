#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MoE 门控选择一致率验证。

dit_step.onnx 有第二个输出 gate_idx（门控选中的专家编号）。本脚本用几条
不同方言的文本构造相同输入，分别跑 FP32 与各 INT8 图，比较 gate_idx
与 FP32 是否一致——这是“敏感层保护”最直接的证据（门控选错专家=选错方言）。

用法：python gate_consistency.py [--models ../models]
"""

import argparse
import os
import tempfile

# quantize_dynamic 会把整图写临时目录；/tmp 只有 512MB，必须指到磁盘，
# 且要在 import onnxruntime / 首次用 tempfile 之前定下来。
_TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp_on_disk")
os.makedirs(_TMP, exist_ok=True)
tempfile.tempdir = os.path.abspath(_TMP)
os.environ["TMPDIR"] = os.path.abspath(_TMP)

import numpy as np
import onnxruntime as ort

from baseline import Tokenizer

SR_MEL = 100


def load_prompts(repo_dir):
    """从官方 prompts 里取几条方言的 IPA 文本：[(方言名, ref_ipa)]"""
    out = []
    pdir = os.path.join(repo_dir, "prompts")
    for name in sorted(os.listdir(pdir)):
        if name.endswith("_prompt.txt"):
            with open(os.path.join(pdir, name), encoding="utf-8") as f:
                parts = f.read().strip().split("\t")
            if len(parts) >= 3:
                out.append((name.replace("_prompt.txt", ""), parts[2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "DiaMoE-TTS"))
    args = ap.parse_args()
    models = os.path.abspath(args.models)

    tok = Tokenizer(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "DiaMoE-TTS", "diamoe_tts", "data", "vocab.txt"))

    prompts = load_prompts(args.repo)[:8]

    # 为每条方言构造一组固定输入（同 seed 噪声）。目标文本就用该方言自己的
    # IPA 串（不同方言的文本内容不同，门控才有区分度）；时长贴近文本长度，
    # 避免 padding 把 pooled 门控信号稀释掉（真实长句推理同理）。
    cases = []
    for i, (dialect, ref_ipa) in enumerate(prompts):
        ref_ids = tok.encode(ref_ipa)
        gen_ids = tok.encode(ref_ipa)  # 用本方言文本当生成目标
        full = ref_ids + [0] + gen_ids
        gate = [-1] * (len(ref_ids) + 1) + gen_ids
        duration = min(384, max(64, (len(full) + 15) // 16 * 16))

        def pad(ids, duration=duration):
            v = np.full((1, duration), -1, dtype=np.int64)
            v[0, :min(len(ids), duration)] = ids[:duration]
            return v

        rng = np.random.default_rng(1000 + i)
        feed = {
            "x": rng.standard_normal((1, duration, SR_MEL), dtype=np.float32),
            "cond": np.zeros((1, duration, SR_MEL), dtype=np.float32),
            "text": pad(full), "text_gate": pad(gate),
            "time": np.array([0.4], dtype=np.float32),
            "moe_flag": np.array([1.0], dtype=np.float32),
            "positions": np.arange(duration, dtype=np.int64),
        }
        cases.append((dialect, feed))
    print(f"测试方言数: {len(cases)}")

    # 旧方案的两张图（dit_step_int8_pertensor/perchannel.onnx）是在导出图加
    # gate_idx 输出之前生成的，没有第二个输出，测不了门控。这里用现 FP32 图
    # 按旧方案（无敏感层保护）临时再量化两份带 gate 输出的等价图做对比，
    # 文件以 _tmp_ 开头，不随包发布。
    from onnxruntime.quantization import quantize_dynamic, QuantType

    fp32 = os.path.join(models, "dit_step.onnx")
    old_graphs = {}
    for tag, pc in (("旧per-tensor无保护", False), ("旧per-channel无保护", True)):
        tmp = os.path.join(models, f"_tmp_int8_{'pc' if pc else 'pt'}_old.onnx")
        if not os.path.exists(tmp):
            quantize_dynamic(model_input=fp32, model_output=tmp,
                             weight_type=QuantType.QInt8,
                             op_types_to_quantize=["MatMul", "Gemm"], per_channel=pc)
        old_graphs[tag] = tmp

    graphs = (["dit_step.onnx"] + list(old_graphs.values())
              + ["dit_step_int8.onnx",
                 "dit_step_int8_protectA.onnx",
                 "dit_step_int8_adaln16.onnx",
                 "dit_step_int8_protectA_adaln16.onnx",
                 "dit_step_int8_protectAp_ff16.onnx",
                 "dit_step_int8_protectAp_ff16_adaln16.onnx"])
    labels = ["FP32", "旧per-tensor无保护", "旧per-channel无保护",
              "INT8无保护(原定稿,per-channel)",
              "INT8+protectA", "INT8+adaln16", "INT8+protectA+adaln16",
              "INT8+protectAp+ff16", "INT8+protectAp+ff16+adaln16"]
    ref_choices = None
    for label, g in zip(labels, graphs):
        path = g if os.path.isabs(g) else os.path.join(models, g)
        if not os.path.exists(path):
            print(f"{label}: 图不存在，跳过")
            continue
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        choices = []
        for dialect, feed in cases:
            out = sess.run(None, feed)
            choices.append(int(out[1][0]))
        del sess
        if ref_choices is None:
            ref_choices = choices
            print(f"{label} (参考): {choices}")
        else:
            agree = sum(int(a == b) for a, b in zip(choices, ref_choices))
            print(f"{label}: 选择 {choices}，与 FP32 一致 {agree}/{len(choices)} = {agree / len(choices):.1%}")


if __name__ == "__main__":
    main()
