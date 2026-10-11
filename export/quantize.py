#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DiaMoE-TTS 量化脚本（可复跑，分阶段、注意内存）。

产出（models/）：
  dit_step_fp16.onnx    FP16：PyTorch 侧 half 后重导出（图内半精度、IO 保持 FP32）—— GPU 推荐
  vocos_spec_fp16.onnx  Vocos FP16
  dit_step_int8.onnx    INT8：onnxruntime quantize_dynamic 对 FP32 图做 MatMul/Gemm 权重
                        per-channel 动态量化（每输出通道一组 scale）—— CPU / 体积优先
                        （门控、embedding、norm 等小层天然不在动态量化范围内；
                        旧 per-tensor 版会备份为 dit_step_int8_pertensor.onnx 仅供误差对比）
  （preprocess.onnx 仅 4.4MB 且含 log，保持 FP32）
  保护档与对照图（int8 阶段附带生成，失败只打印不阻塞；三方验证结论见
  README 第 2 节）：
    dit_step_int8_protectA.onnx           A 组小层整节点保 FP32 —— CPU 默认推荐档
    dit_step_int8_fullprotect.onnx        A 组 + attn_norm 全保 FP32 —— 上限对照
    dit_step_int8_adaln16.onnx           仅 attn_norm(AdaLN) 保 FP16 —— OPTIMIZATION.md (b) 折中
    dit_step_int8_protectA_adaln16.onnx  A 组保 FP32 + attn_norm 保 FP16 的叠加版
    dit_step_int8_protectAp_ff16.onnx    FFN 组件组保 FP16 + (A组+input_embed) 保 FP32
                                         —— 敏感度扫描表驱动的精度档（FFN 是最大单步误差源）
    dit_step_int8_protectAp_ff16_adaln16.onnx  全组合：ff→FP16 + attn_norm→FP16 + A⁺→FP32

为什么 FP16 不用 onnxconverter-common 的图手术：1.4GB 单文件图做 protobuf 级
转换在本机（7GB RAM）会 OOM；half 后从 PyTorch 重导出内存可控且图更干净。

用法（分阶段跑，避免同时加载多个大模型）：
  python quantize.py fp16   --ckpt ../weights/10ep_mlpEXP_9.pt --models ../models
  python quantize.py int8   --models ../models
  python quantize.py errors --models ../models     # 单步误差 vs FP32
  python quantize.py sizes  --models ../models     # 体积汇总
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def size_mb(path):
    return os.path.getsize(path) / 1e6


def do_fp16(ckpt, models):
    import torch
    from export_models import DiTStep, VocosSpec, build_model

    torch.manual_seed(0)
    model = build_model(ckpt, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "DiaMoE-TTS", "diamoe_tts", "data", "vocab.txt"))
    dit = model.transformer.half()
    step = DiTStep(dit).eval()
    n = 256
    dummy = (
        torch.randn(1, n, 100), torch.randn(1, n, 100),
        torch.randint(0, 100, (1, n), dtype=torch.int64),
        torch.randint(0, 100, (1, n), dtype=torch.int64),
        torch.tensor([0.5]), torch.tensor([1.0]),
        torch.arange(n, dtype=torch.int64),
    )
    dst = os.path.join(models, "dit_step_fp16.onnx")
    with torch.no_grad():
        torch.onnx.export(
            step, dummy, dst,
            input_names=["x", "cond", "text", "text_gate", "time", "moe_flag", "positions"],
            output_names=["vel", "gate_idx"],
            dynamic_axes={"x": {1: "N"}, "cond": {1: "N"}, "text": {1: "N"},
                          "text_gate": {1: "N"}, "positions": {0: "N"}, "vel": {1: "N"}},
            opset_version=17, dynamo=False, do_constant_folding=True,
        )
    print(f"dit_step_fp16.onnx: {size_mb(dst):.1f} MB")
    del model, step, dit

    # Vocos FP16
    from vocos import Vocos

    vocos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights", "vocos")
    vocoder = Vocos.from_hparams(os.path.join(vocos_dir, "config.yaml"))
    vsd = torch.load(os.path.join(vocos_dir, "pytorch_model.bin"),
                     map_location="cpu", weights_only=True)
    vocoder.load_state_dict(vsd)
    del vsd
    spec_model = VocosSpec(vocoder.eval().half()).eval()
    mel_dummy = torch.randn(1, 100, 64)
    dst = os.path.join(models, "vocos_spec_fp16.onnx")
    with torch.no_grad():
        torch.onnx.export(
            spec_model, (mel_dummy,), dst,
            input_names=["mel"], output_names=["spec"],
            dynamic_axes={"mel": {2: "N"}, "spec": {2: "N"}},
            opset_version=17, dynamo=False, do_constant_folding=True,
        )
    print(f"vocos_spec_fp16.onnx: {size_mb(dst):.1f} MB")


def do_int8(models):
    # 量化器会把整图以 external-data 形式写到临时目录再做 shape inference；
    # 本机 /tmp 只有 512MB，必须把临时目录指到磁盘上（复跑时同理）。
    import tempfile

    tmp_dir = os.path.join(os.path.dirname(models), "tmp_on_disk")
    os.makedirs(tmp_dir, exist_ok=True)
    tempfile.tempdir = tmp_dir
    os.environ["TMPDIR"] = tmp_dir

    from onnxruntime.quantization import QuantType, quantize_dynamic

    src = os.path.join(models, "dit_step.onnx")
    dst = os.path.join(models, "dit_step_int8.onnx")
    # 只量化 MatMul/Gemm：卷积保持 FP32。一是 text/input 里的 depthwise 卷积参数量很小；
    # 二是 ConvInteger 在部分 ORT 构建（如 Rust ort 的预编译二进制）里没有 CPU 内核，会加载失败。
    # per_channel=True：每个输出通道一组 scale/zero-point，比 per-tensor 精度更好，体积略增。
    #
    # 敏感层保护（保持高精度，不量化）的分档结论见 OPTIMIZATION.md (b)(d) 实测：
    #   A 组（真小层，保 FP32 几乎不花体积，实测仅 +10.5MB，且是门控 8/8 的关键）：
    #     - moe/gate        门控 Linear：选错专家 = 选错方言，路由必须精准
    #     - time_embed      时间步 MLP：调制信号来源，体积小、影响全局
    #     - norm_out        输出前最后的调制 Linear（1024→6144，一层）
    #     - proj_out        最终投影到 mel：直接决定输出幅度
    #   B 组（体积代价太大，不能全保 FP32）：
    #     - attn_norm  每层 AdaLN 调制 Linear（1024→6144），22 层共约 138M 参数，
    #       占全模型近 40%。全保 FP32 会让 INT8 从约 394MB 膨胀到约 819MB，
    #       折中是保 FP16（实测约 +137MB，见下方实验分支）。
    # 基础图（dit_step_int8.onnx，最小体积档）不做保护：只用已充分验证的
    # per-channel 动态量化；推荐保护档（protectA / protectAp_ff16 / combo）
    # 由下方实验分支从基础图手术生成。基础图里的大头（注意力 QKV/投影、FFN、
    # 专家 MLP 本体、AdaLN）保持 INT8 per-channel。
    #
    # 实现方式：quantize_dynamic 的 nodes_to_exclude 在 ORT 1.31 实测不可靠——
    # 量化前的图优化会把 Gemm 改名（…/Gemm → …/Gemm_MatMul_quant），排除名单对不上，
    # 敏感层仍被量化（复查发现 26 处漏网）。因此改用"量化后图手术"：
    # 全量 INT8 后，把敏感节点的量化子图（DynamicQuantizeLinear→MatMulInteger→
    # Cast→Mul→Add）整段替换回 FP32 图里的原 MatMul/Gemm 节点与原始权重。
    # 术后按节点名复查：敏感节点不得再是 MatMulInteger，且 ORT 必须能加载。
    import onnx as _onnx

    patterns_a = ("/moe/gate/", "/norm_out/", "/proj_out", "/time_embed/")
    patterns_ab = patterns_a + ("/attn_norm/",)

    # 先把上一版 INT8 备份下来，供 errors 阶段并排对比（不随包发布）。
    for backup_name in ("dit_step_int8_pertensor.onnx", "dit_step_int8_perchannel.onnx"):
        backup = os.path.join(models, backup_name)
        if os.path.exists(dst) and not os.path.exists(backup):
            os.replace(dst, backup)
            print(f"上一版 INT8 已备份为 {backup_name}（仅供对比）")
            break

    quantize_dynamic(model_input=src, model_output=dst, weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"], per_channel=True)
    # 基础图（上一轮定稿、现最小体积档）：per-channel 动态量化、无敏感层保护。
    # 已实测 ORT 可加载、单步 cos 0.99880、Python 端到端基线与 Rust 同结构图
    # 冒烟均通过。
    print(f"dit_step_int8.onnx（per-channel 动态量化，最小体积档）: {size_mb(dst):.1f} MB")

    # 保护档与对照图：敏感层保护图手术，从上面的基础图逐档生成。
    # protectA/fullprotect 在上一轮已通过节点级复查（敏感节点可还原、
    # MatMulInteger 漏网 0）；adaln16 / protectA_adaln16 / protectAp_ff16 /
    # combo 是本轮按 OPTIMIZATION.md (b) 与敏感度扫描新增：attn_norm（AdaLN
    # 调制 Linear，22 层约 138M 参数、占全模型近 40%）不保 FP32（+约 415MB
    # 太贵）而保 FP16（实测约 +137MB）；FFN 组件组保 FP16（扫描实测它是最大
    # 单步误差源）。各图均已过 quantize.py errors、gate_consistency.py、
    # compare_e2e.py 三方验证（README 第 2 节）；单张手术失败不影响基础图。
    import shutil

    def try_exp(tag, surgeries):
        """surgeries: [(patterns, fp16), ...]，按序对同一文件逐个执行手术。"""
        exp = os.path.join(models, tag)
        try:
            shutil.copyfile(dst, exp)
            for pats, fp16 in surgeries:
                restore_sensitive(exp, src, pats, fp16=fp16)
            print(f"实验版 {tag}: {size_mb(exp):.1f} MB（重生成后请复跑 errors/gate/e2e 验证）")
        except Exception as e:  # 单张手术失败不阻塞基础图产出
            print(f"实验版 {tag} 手术失败（不影响基础图）: {type(e).__name__}: {e}")
            if os.path.exists(exp):
                os.remove(exp)

    try_exp("dit_step_int8_protectA.onnx", [(patterns_a, False)])
    try_exp("dit_step_int8_fullprotect.onnx", [(patterns_ab, False)])
    # 仅 AdaLN 保 FP16（实测约 +137MB）
    try_exp("dit_step_int8_adaln16.onnx", [(("/attn_norm/",), True)])
    # A 组保 FP32 + AdaLN 保 FP16：两次手术 patterns 不相交、各自 load→改→save，互不干扰
    try_exp("dit_step_int8_protectA_adaln16.onnx",
            [(("/attn_norm/",), True), (patterns_a, False)])
    # 敏感度扫描表驱动的精度档：FFN 组件组（block:ff，全部 22 层的 ff，约 92.3M 参数）
    # 保 FP16（扫描实测它是最大单步误差源，Δmean +0.01674），叠加 A⁺组 = A 组 +
    # input_embed（+0.7MB 买 +0.00377 的高性价比保护）保 FP32。patterns 来自
    # sensitivity_scan.py 的组发现（transformer_blocks.{i}/ff/，22 = DiT depth）。
    patterns_ff16 = tuple(f"transformer_blocks.{i}/ff/" for i in range(22))
    patterns_aplus = patterns_a + ("/input_embed/",)
    try_exp("dit_step_int8_protectAp_ff16.onnx",
            [(patterns_ff16, True), (patterns_aplus, False)])
    # 全组合：FFN→FP16 + AdaLN(attn_norm)→FP16 + A⁺→FP32，三段手术 patterns
    # 互不相交、顺序执行（单步误差与端到端质量并重，供与 protectAp_ff16 对照）
    try_exp("dit_step_int8_protectAp_ff16_adaln16.onnx",
            [(patterns_ff16, True), (("/attn_norm/",), True), (patterns_aplus, False)])


def restore_sensitive(qpath, fpath, patterns, fp16=False):
    """把量化图 qpath 里敏感节点的 INT8 子图还原（落盘版，load→手术→save→复查）。

    fp16=False：整节点还原为 FP32（原节点 + FP32 原权重直接落图）。
    fp16=True ：敏感 MatMul/Gemm 以 FP16 计算（OPTIMIZATION.md (b) 的 AdaLN 折中）：
      Cast(输入→F16) → MatMul/Gemm(F16 权重) → Cast(输出→F32)。输出名保持原名，
      下游消费者不用改；权重以新名（原名+"_fp16"）落图，Gemm 的 bias 也转 F16
      （Gemm 内部加法要求同 dtype）。
    """
    import onnx as _onnx

    fmodel = _onnx.load(fpath, load_external_data=False)
    qmodel = _onnx.load(qpath)  # 需要改 initializer，整图载入
    st = restore_sensitive_inplace(qmodel, fmodel, patterns, fp16=fp16)
    _onnx.save(qmodel, qpath)
    print(f"敏感层手术完成: 还原 {st['restored']} 个节点"
          f"（{'FP16' if fp16 else 'FP32'}），本来就是 FP32 的 {st['already_fp32']} 个，"
          f"清掉无用 initializer {st['dropped_inits']} 个")
    del qmodel

    # 复查：按最终图的节点名统计，敏感节点不得再出现 MatMulInteger
    chk = _onnx.load(qpath, load_external_data=False)
    mi_names = [n.name for n in chk.graph.node if n.op_type == "MatMulInteger"]
    leaked = [n for n in mi_names if any(p in n for p in patterns)]
    sens_fp32 = sum(1 for n in chk.graph.node
                    if n.op_type in ("MatMul", "Gemm") and any(p in n.name for p in patterns))
    print(f"复查: MatMulInteger 共 {len(mi_names)} 个，敏感漏网 {len(leaked)} 个；"
          f"敏感 MatMul/Gemm 以{'FP16 还原' if fp16 else 'FP32'}存在 {sens_fp32} 个")
    del chk, fmodel


def restore_sensitive_inplace(qmodel, fmodel, patterns, fp16=False):
    """restore_sensitive 的核心：对内存中的图对象做手术，不落盘。

    供 sensitivity_scan.py 复用（同一份 fmodel 反复手术不同的 INT8 副本，
    省掉每项几百 MB 的重复 IO）。qmodel 会被就地修改；fmodel 只读。
    返回统计 dict；量化链结构异常立即 raise RuntimeError（带节点上下文，
    不静默吞）。手术逻辑（链定位/删除）FP32 与 FP16 路径完全共用，区别只在
    链尾位置插入的替换节点：FP32 是 1 个原节点，FP16 是 Cast→MM/Gemm→Cast 3 个。
    """
    import numpy as np
    import onnx as _onnx
    from onnx import TensorProto, numpy_helper

    fg, qg = fmodel.graph, qmodel.graph

    sensitive = [n for n in fg.node
                 if n.op_type in ("MatMul", "Gemm")
                 and any(p in n.name for p in patterns)]
    if not sensitive:
        raise RuntimeError(f"FP32 图里没有任何匹配 {patterns} 的 MatMul/Gemm 节点，patterns 写错？")
    f_inits = {i.name: i for i in fg.initializer}

    producer = {}
    for n in qg.node:
        for o in n.output:
            producer[o] = n
    consumers = {}
    for n in qg.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    node_by_name = {n.name: n for n in qg.node}
    # 量化脚手架算子白名单：既是 collect_exclusive_producers 的"可吸收"判据，
    # 也是术后结构性校验（安全网）的合法 op 集合。
    scaffold_ops = {"DynamicQuantizeLinear", "MatMulInteger", "Cast", "Mul",
                    "Add", "Div", "Sub", "Reshape", "Unsqueeze", "Squeeze",
                    "Concat", "Constant", "DequantizeLinear"}
    # FP16 手术会新造张量/initializer 名（_fp16cast / _fp16out / _fp16 后缀），做唯一性防御
    existing_names = set(producer) | {i.name for i in qg.initializer}
    existing_names.update(i.name for i in qg.input)
    existing_names.update(o.name for o in qg.output)
    to_delete = set()
    inserts = []  # (链尾节点名, 按拓扑序的替换节点列表)
    fp16_inits = {}  # 新权重名 -> FP16 TensorProto（fp16 路径，手术完统一落图）
    cast_out_cache = {}  # 输入张量名 -> 已创建的 FP16 Cast 输出名（共享输入时复用，避免重复张量定义）
    restored, already_fp32 = 0, 0

    for s in sensitive:
        # Gemm 源节点的量化节点叫 <name>_MatMul_quant，MatMul 源节点叫 <name>_quant
        mi = (node_by_name.get(s.name + "_MatMul_quant")
              or node_by_name.get(s.name + "_quant"))
        if mi is None:
            already_fp32 += 1  # 量化器没有动它，保持 FP32
            continue
        # 收集量化子图，分两步：
        # (a) 从 MatMulInteger 的输出沿单一消费者链向下走，到原输出名为止。
        #     注意不能沿输入向上做闭包扩展：一份 x 的 DynamicQuantizeLinear
        #     会被几十个 MatMul 共享（如 DiT 隐藏状态），横向扩展会误吞全图。
        # (b) 对子图节点的每个输入，向上递归收集"只服务于本子图"的生产者
        #     （scale 组合节点、专属 DynamicQuantizeLinear 等），删干净，
        #     不留悬空引用（第一版手术就栽在 scale 张量悬空、图损坏）。
        group = {mi.name}
        cur = mi.output[0]
        tail_name = None
        while cur != s.output[0]:
            nxt = consumers.get(cur, [])
            if len(nxt) != 1:
                raise RuntimeError(f"{s.name}: 量化链在 {cur} 处消费者数 {len(nxt)}，手术中止")
            group.add(nxt[0].name)
            cur = nxt[0].output[0]
            tail_name = nxt[0].name
        if tail_name is None:
            raise RuntimeError(f"{s.name}: 量化链为空，手术中止")

        def collect_exclusive_producers(node_names):
            """把只被 node_names 内部节点消费的量化脚手架生产者递归并进来。

            两条吸收红线（踩中任一条都跳过、且不再沿该路径向上扩展）：
            1. 生产者不是脚手架算子（Sin/Cos 等真实计算）——真实计算即使只服务于
               本子图也不吸收（time_embed 的激活就由 Sin/Cos 独家生产，吸收后白
               名单校验会整组中止，还原节点输入也会悬空）；
            2. 生产者的输出里有敏感节点 s 的直接输入——还原的 FP32/FP16 节点要
               重新消费这些张量名（FP16 路径的输入 Cast 也锚在 s.input[0] 上），
               其生产者必须存活。
            被保留的节点若因此变成 0 消费者（如只喂 DQL 的 Concat/Sin/Cos），
            属于死代码，ONNX 合法（ORT 加载时自动剪枝），不必特判。
            """
            anchor_inputs = set(s.input)
            work = [node_by_name[nm] for nm in list(node_names)]
            added = set()
            while work:
                n = work.pop()
                for i in n.input:
                    if not i:
                        continue
                    p = producer.get(i)
                    if p is None or p.name in group or p.name in added:
                        continue
                    if p.op_type not in scaffold_ops:
                        continue  # 红线 1：真实计算不吸收、不向上扩展
                    if any(o in anchor_inputs for o in p.output):
                        continue  # 红线 2：还原节点的输入锚点必须存活
                    # p 的全部输出是否只被（group ∪ added）消费？
                    p_outs = list(p.output)
                    only_inside = all(
                        all(c.name in group or c.name in added
                            for c in consumers.get(o, []))
                        for o in p_outs)
                    if only_inside and consumers.get(p_outs[0], []):
                        added.add(p.name)
                        work.append(p)
            return added

        group |= collect_exclusive_producers(group)
        # 结构性校验安全网（吸收红线理论上已挡住非脚手架节点，这里兜底）：
        # 子图里只能有量化脚手架 op，且恰好含 1 个 MatMulInteger；出现第二个
        # 整数 MatMul 或 Gemm/Conv 等真实计算，说明误吞了下游，立即中止。
        bad_ops = [f"{n.op_type}:{n.name}" for n in qg.node
                   if n.name in group and n.op_type not in scaffold_ops]
        n_mi = sum(1 for n in qg.node
                   if n.name in group and n.op_type == "MatMulInteger")
        if bad_ops or n_mi != 1:
            raise RuntimeError(f"{s.name}: 子图结构异常 bad={bad_ops[:3]} MatMulInteger数={n_mi}，手术中止")
        # 安全校验：组内节点的输出不得被组外节点消费（原输出名除外，它会被还原节点重新产生）
        group_outputs = set()
        for n in qg.node:
            if n.name in group:
                group_outputs.update(n.output)
        for out in group_outputs:
            if out == s.output[0]:
                continue
            for c in consumers.get(out, []):
                if c.name not in group:
                    raise RuntimeError(f"{s.name}: 子图输出 {out} 还被组外 {c.name} 消费，手术中止")
        tail_name = next(n.name for n in qg.node
                         if n.name in group and s.output[0] in n.output)
        to_delete.update(group)

        # 构造链尾位置的替换节点（保持拓扑顺序：castin → matmul → castout）
        if not fp16:
            # FP32：直接放回原节点
            new_node = _onnx.NodeProto()
            new_node.CopyFrom(s)
            replace_nodes = [new_node]
        else:
            # FP16：Cast(输入→F16) → MatMul/Gemm(F16 权重) → Cast(输出→F32)。
            # 输出 Cast 沿用原输出名 s.output[0]，下游不用改；多个敏感节点共享
            # 同一输入（如 9 个专家共享 text_embed）时复用同一个输入 Cast，
            # 避免两个节点写同一张量名。s.input[0] 可能是 initializer 也可能
            # 是动态张量，输出名加后缀保证唯一即可。
            src = s.input[0]
            pre = []
            if src in cast_out_cache:
                cast_out = cast_out_cache[src]
            else:
                cast_out = src + "_fp16cast"
                if cast_out in existing_names:
                    raise RuntimeError(f"{s.name}: FP16 Cast 输出名 {cast_out} 已被占用，手术中止")
                existing_names.add(cast_out)
                pre = [_onnx.helper.make_node(
                    "Cast", [src], [cast_out], name=s.name + "_castin",
                    to=TensorProto.FLOAT16)]
                cast_out_cache[src] = cast_out
            new_node = _onnx.NodeProto()
            new_node.CopyFrom(s)
            in_list = [cast_out]
            for inp in s.input[1:]:
                if inp not in f_inits:
                    raise RuntimeError(
                        f"{s.name}: 权重输入 {inp} 不是 FP32 图的 initializer，无法预转 FP16，手术中止")
                wname = inp + "_fp16"
                if wname not in fp16_inits:
                    if wname in existing_names:
                        raise RuntimeError(f"{s.name}: FP16 权重名 {wname} 已被占用，手术中止")
                    arr = numpy_helper.to_array(f_inits[inp]).astype(np.float16)
                    fp16_inits[wname] = numpy_helper.from_array(arr, wname)
                    existing_names.add(wname)
                in_list.append(wname)  # Gemm 的 bias(C) 也转 F16：内部加法要求同 dtype
            del new_node.input[:]
            new_node.input.extend(in_list)
            mid_out = s.output[0] + "_fp16out"  # 不能直接用原输出名，后面还要 Cast 回 FP32
            if mid_out in existing_names:
                raise RuntimeError(f"{s.name}: FP16 中间输出名 {mid_out} 已被占用，手术中止")
            existing_names.add(mid_out)
            del new_node.output[:]
            new_node.output.extend([mid_out])
            post = [_onnx.helper.make_node(
                "Cast", [mid_out], [s.output[0]], name=s.name + "_castout",
                to=TensorProto.FLOAT)]
            replace_nodes = pre + [new_node] + post
        inserts.append((tail_name, replace_nodes))
        restored += 1

    # 重建节点表：删除链节点，在链尾原位置插入替换节点
    # （一个位置可能插多个节点：FP16 路径是 Cast → MatMul/Gemm → Cast）
    insert_at = {}
    for tail_name, nodes in inserts:
        insert_at.setdefault(tail_name, []).extend(nodes)
    new_nodes = []
    for n in qg.node:
        if n.name in insert_at:
            new_nodes.extend(insert_at[n.name])
        if n.name in to_delete:
            continue
        new_nodes.append(n)
    del qg.node[:]
    qg.node.extend(new_nodes)

    # 拷贝还原节点要用的权重（量化子图引用的是 *_quantized 版本，原名在量化图里已不存在）：
    # FP32 路径拷原名 FP32；FP16 路径拷 _fp16 新名（Gemm 的 bias 已同转 FP16）。
    if fp16:
        qg.initializer.extend(tp for _, tp in sorted(fp16_inits.items()))
    else:
        q_init_names = {i.name for i in qg.initializer}
        for s in sensitive:
            for inp in s.input[1:]:
                if inp in f_inits and inp not in q_init_names:
                    cp = _onnx.TensorProto()
                    cp.CopyFrom(f_inits[inp])
                    qg.initializer.append(cp)
                    q_init_names.add(inp)

    # 清除不再被引用的 initializer（*_quantized / zero_point / scales 等）。
    # fp16 路径只按 used 清理：Gemm 的 bias 等原名 FP32 权重已被 _fp16 新名替代、
    # 无节点引用，不能因"是敏感节点的原输入"而保留（否则 22 层 attn_norm 的
    # weight/bias 会成为死 initializer，ORT 加载时报 CleanUnusedInitializers 警告）。
    used = set()
    for n in qg.node:
        used.update(n.input)
    used.update(o.name for o in qg.output)
    if fp16:
        keep = [i for i in qg.initializer if i.name in used]
    else:
        keep = [i for i in qg.initializer if i.name in used or i.name in f_inits and any(
            i.name in s.input for s in sensitive)]
    kept_names = {i.name for i in keep}
    dropped = [i.name for i in qg.initializer if i.name not in kept_names]
    del qg.initializer[:]
    qg.initializer.extend(keep)

    return {"restored": restored, "already_fp32": already_fp32,
            "dropped_inits": len(dropped), "fp16": fp16}


def do_errors(models):
    import numpy as np
    import onnxruntime as ort
    import torch

    ref = torch.load(os.path.join(models, "dit_step_ref.pt"), weights_only=False)
    names = ["x", "cond", "text", "text_gate", "time", "moe_flag", "positions"]
    feed = {n: v for n, v in zip(names, ref["inputs"])}

    def run(path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        out = sess.run(None, feed)[0]
        del sess
        return out

    base = run(os.path.join(models, "dit_step.onnx")).astype(np.float64).ravel()
    for tag in ("dit_step_fp16.onnx", "dit_step_int8_pertensor.onnx",
                "dit_step_int8_perchannel.onnx", "dit_step_int8.onnx",
                "dit_step_int8_protectA.onnx", "dit_step_int8_adaln16.onnx",
                "dit_step_int8_protectA_adaln16.onnx", "dit_step_int8_protectAp_ff16.onnx",
                "dit_step_int8_protectAp_ff16_adaln16.onnx", "dit_step_int8_fullprotect.onnx"):
        path = os.path.join(models, tag)
        if not os.path.exists(path):
            print(f"{tag}: 不存在，跳过")
            continue
        b = run(path).astype(np.float64).ravel()
        err = {
            "max_abs": float(np.max(np.abs(base - b))),
            "mean_abs": float(np.mean(np.abs(base - b))),
            "cos": float(base @ b / (np.linalg.norm(base) * np.linalg.norm(b) + 1e-12)),
        }
        print(f"{tag} 单步误差 vs FP32: {err}")


def do_sizes(models):
    print("== 体积汇总 (MB) ==")
    for f in sorted(os.listdir(models)):
        if f.endswith(".onnx"):
            print(f"  {f}: {size_mb(os.path.join(models, f)):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fp16", "int8", "errors", "sizes"])
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "..", "weights", "10ep_mlpEXP_9.pt"))
    ap.add_argument("--models", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
    args = ap.parse_args()
    models = os.path.abspath(args.models)

    if args.stage == "fp16":
        do_fp16(args.ckpt, models)
    elif args.stage == "int8":
        do_int8(models)
    elif args.stage == "errors":
        do_errors(models)
    else:
        do_sizes(models)


if __name__ == "__main__":
    main()
