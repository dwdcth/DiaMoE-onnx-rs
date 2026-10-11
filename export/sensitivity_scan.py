#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按层/按组件敏感度扫描（OPTIMIZATION.md (d) 第二条：按层做敏感度扫描）。

目的：在定稿 INT8 图（per-channel 动态量化、无保护）的基础上，逐组把某组
MatMul/Gemm 还原为 FP16，用固定参考输入（dit_step_ref.pt）测单步输出 vs
FP32 基线的误差下降，产出「误差收益/体积代价」排序表，为选择保护清单
（贪心加保护）提供数据。对应 OPTIMIZATION.md 建议执行顺序第 3 条。

扫描项（组从 FP32 图实际发现，不猜节点名，分类映射表会打印存档）：
  - block.{i}        22 个 transformer_blocks 层的整层组（层内全部 MatMul/Gemm）
  - block:{comp}     跨层同组件组（如全部 attn_norm / attn / ff ...，以实际名称为准）
  - {首段名}          transformer_blocks 之外的组（text_blocks / input_embed /
                     moe(experts+gate) / time_embed / norm_out / proj_out 等）

每组流程：deepcopy 基础 INT8 图 → restore_sensitive_inplace(fp16=True) →
SerializeToString → ort.InferenceSession(bytes)（不落盘，省 IO）→ 跑固定输入
→ 记录 max_abs/mean_abs/cos 与该组参数量（体积代价 ≈ 参数量×1 字节：
FP16 2B - INT8 ~1B）→ 逐项 del 释放内存。

内存预算：FP32 图 proto（~1.4GB）+ INT8 基础图 + 每项副本（~400MB）+
ORT session（~1-2GB）；30GB RAM 足够，但必须逐项释放（fmodel 全程只加载一次）。

用法（在 export/ 下）：
  ../.venv/bin/python sensitivity_scan.py --models ../models --threads 8
  可选：--fp32/--int8/--ref 换文件名，--out 换 JSON 输出路径
依赖：export_models.py 产出的 dit_step.onnx + dit_step_ref.pt，
      quantize.py int8 阶段产出的 dit_step_int8.onnx。
基线说明：误差基准用 dit_step_ref.pt 里存的 PyTorch FP32 输出（ORT FP32 图
与它的单步差 <1e-5，见 README 第 1 节），省去每轮加载 1.4GB FP32 session。
"""

import argparse
import json
import os
import re
import sys
import tempfile

# 与 gate_consistency.py 同理：ORT / 量化相关临时文件必须指到磁盘（/tmp 只有 512MB），
# 且要在首次 import onnxruntime / 用 tempfile 之前定下来。
_TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp_on_disk")
os.makedirs(_TMP, exist_ok=True)
tempfile.tempdir = os.path.abspath(_TMP)
os.environ.setdefault("TMPDIR", os.path.abspath(_TMP))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BLOCK_RE = re.compile(r"transformer_blocks\.(\d+)/")


def discover_groups(fg):
    """从 FP32 图实际发现全部 MatMul/Gemm 节点并分组。

    返回 [(组名, patterns 元组, [节点名列表]), ...]：
      block.{i}    —— 第 i 层整层组，pattern 为 "transformer_blocks.{i}/"
      block:{comp} —— 跨层组件组，pattern 为各层的 "transformer_blocks.{i}/{comp}/"
      {首段名}      —— blocks 之外的组，pattern 为 "/{首段}/"（moe 组天然含 experts+gate）
    """
    blocks = {}   # i -> {comp: [节点名]}
    outside = {}  # 首段名 -> [节点名]
    for n in fg.node:
        if n.op_type not in ("MatMul", "Gemm"):
            continue
        m = BLOCK_RE.search(n.name)
        if m:
            i = int(m.group(1))
            comp = n.name[m.end():].split("/")[0]
            blocks.setdefault(i, {}).setdefault(comp, []).append(n.name)
        else:
            seg = n.name.lstrip("/").split("/")[0]
            outside.setdefault(seg, []).append(n.name)

    groups = []
    for i in sorted(blocks):
        names = [nm for comp in blocks[i] for nm in blocks[i][comp]]
        groups.append((f"block.{i}", (f"transformer_blocks.{i}/",), names))
    for comp in sorted({c for d in blocks.values() for c in d}):
        pats = tuple(f"transformer_blocks.{i}/{comp}/"
                     for i in sorted(blocks) if comp in blocks[i])
        names = [nm for i in sorted(blocks) for nm in blocks[i].get(comp, [])]
        groups.append((f"block:{comp}", pats, names))
    for seg in sorted(outside):
        groups.append((seg, (f"/{seg}/",), outside[seg]))
    return groups


def main():
    ap = argparse.ArgumentParser(description="按层/按组件敏感度扫描（INT8 图逐组还原 FP16）")
    ap.add_argument("--models", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "..", "models"))
    ap.add_argument("--fp32", default="dit_step.onnx")
    ap.add_argument("--int8", default="dit_step_int8.onnx")
    ap.add_argument("--ref", default="dit_step_ref.pt")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="结果 JSON 路径，默认 {models}/sensitivity_scan.json")
    args = ap.parse_args()
    models = os.path.abspath(args.models)

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(models, p)

    fp32_path, int8_path, ref_path = resolve(args.fp32), resolve(args.int8), resolve(args.ref)
    for p, tag in ((fp32_path, "FP32 图"), (int8_path, "INT8 基础图"), (ref_path, "参考输入")):
        if not os.path.exists(p):
            raise SystemExit(f"缺少{tag}: {p}（先跑 export_models.py / quantize.py int8）")

    import numpy as np
    import onnx as _onnx
    import onnxruntime as ort
    import torch

    from quantize import restore_sensitive_inplace

    ref = torch.load(ref_path, weights_only=False)
    names = ["x", "cond", "text", "text_gate", "time", "moe_flag", "positions"]
    feed = {n: v for n, v in zip(names, ref["inputs"])}
    ref_out = ref["output"].astype(np.float64).ravel()
    del ref

    print("加载 FP32 图（只读，供手术取原节点/权重，全程仅此一次）...")
    fmodel = _onnx.load(fp32_path, load_external_data=False)
    fg = fmodel.graph
    groups = discover_groups(fg)
    node_by_name = {n.name: n for n in fg.node}
    all_mm_names = [n.name for n in fg.node if n.op_type in ("MatMul", "Gemm")]

    # 组内权重的参数量：用 initializer 的 dims 乘积估算，不物化权重数据
    f_init_sizes = {}
    for i in fg.initializer:
        f_init_sizes[i.name] = int(np.prod(i.dims)) if i.dims else 1

    def group_params(node_names):
        total, seen = 0, set()
        for nm in node_names:
            for inp in node_by_name[nm].input[1:]:
                if inp in f_init_sizes and inp not in seen:
                    seen.add(inp)
                    total += f_init_sizes[inp]
        return total

    print("== 组发现映射表（分类以实际节点名为准，存档用）==")
    for gname, pats, nodes in groups:
        print(f"  {gname}: {len(nodes)} 个 MatMul/Gemm, 参数 {group_params(nodes) / 1e6:.1f}M, "
              f"示例 {nodes[:2]}")

    print("加载 INT8 基础图...")
    base_q = _onnx.load(int8_path)

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads

    def eval_bytes(model_bytes):
        sess = ort.InferenceSession(model_bytes, so, providers=["CPUExecutionProvider"])
        out = sess.run(None, feed)[0]
        del sess
        return out

    def metrics(out):
        b = out.astype(np.float64).ravel()
        d = ref_out - b
        return {
            "max_abs": float(np.max(np.abs(d))),
            "mean_abs": float(np.mean(np.abs(d))),
            "cos": float(ref_out @ b / (np.linalg.norm(ref_out) * np.linalg.norm(b) + 1e-12)),
        }

    print("测基础 INT8 误差基准（不还原任何组）...")
    base_bytes = base_q.SerializeToString()
    base_m = metrics(eval_bytes(base_bytes))
    del base_bytes
    print(f"  INT8 基线: {base_m}")

    results = []
    for gname, pats, nodes in groups:
        # 防御：pattern 子串匹配到的集合必须恰好等于本组节点（防跨组误伤/漏配）
        matched = set(nm for nm in all_mm_names if any(p in nm for p in pats))
        if matched != set(nodes):
            print(f"[警告] {gname}: pattern 匹配 {len(matched)} 个 != 组内 {len(nodes)} 个，"
                  f"跳过该组（需人工核对）")
            results.append({"group": gname, "skipped": True,
                            "matched": len(matched), "expected": len(nodes)})
            continue
        params = group_params(nodes)
        try:
            qcopy = _onnx.ModelProto()  # 每次从 pristine 基础副本开始，避免手术累积
            qcopy.CopyFrom(base_q)
            restore_sensitive_inplace(qcopy, fmodel, pats, fp16=True)
            model_bytes = qcopy.SerializeToString()
            del qcopy
            m = metrics(eval_bytes(model_bytes))
            del model_bytes
        except Exception as e:  # 单组失败不中断整个扫描，但要显式记录（不静默吞）
            print(f"[失败] {gname}: {type(e).__name__}: {e}")
            results.append({"group": gname, "nodes": len(nodes), "params": params,
                            "size_cost_mb": round(params / 1e6, 1),
                            "error": f"{type(e).__name__}: {e}"})
            continue
        row = {"group": gname, "nodes": len(nodes), "params": params,
               "size_cost_mb": round(params / 1e6, 1), **m,
               "delta_mean_abs": base_m["mean_abs"] - m["mean_abs"]}
        results.append(row)
        print(f"  {gname}: n={len(nodes)} params={params / 1e6:.1f}M "
              f"mean_abs={m['mean_abs']:.5f} cos={m['cos']:.6f} "
              f"Δmean={row['delta_mean_abs']:+.5f}")

    # 按 mean_abs 绝对下降量排序（降序）：越靠前的组还原 FP16 收益越大
    ok = [r for r in results if "mean_abs" in r]
    ok.sort(key=lambda r: -r["delta_mean_abs"])
    print(f"\n== 敏感度排序（基础 INT8 mean_abs={base_m['mean_abs']:.5f}，按 Δmean_abs 降序）==")
    print(f"{'组名':<16}{'节点':>5}{'参数(M)':>9}{'+MB':>7}"
          f"{'max_abs':>10}{'mean_abs':>10}{'cos':>10}{'Δmean_abs':>11}")
    for r in ok:
        print(f"{r['group']:<16}{r['nodes']:>5}{r['params'] / 1e6:>9.1f}{r['size_cost_mb']:>7.1f}"
              f"{r['max_abs']:>10.5f}{r['mean_abs']:>10.5f}{r['cos']:>10.6f}"
              f"{r['delta_mean_abs']:>+11.5f}")
    bad = [r for r in results if "mean_abs" not in r]
    if bad:
        print(f"[注意] {len(bad)} 个组被跳过/失败：")
        for r in bad:
            print(f"  {r['group']}: {r.get('error') or 'pattern 匹配数与组内不一致'}")

    out_path = args.out or os.path.join(models, "sensitivity_scan.json")
    payload = {
        "fp32": args.fp32, "int8": args.int8, "ref": args.ref,
        "threads": args.threads, "baseline_int8": base_m,
        "sorted_by_delta": [r["group"] for r in ok],
        "groups": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {out_path}")


if __name__ == "__main__":
    main()
