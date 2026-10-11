#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DiaMoE-TTS -> ONNX 导出脚本（本项目自研导出层，不修改官方仓库代码）。

导出三张图：
  1. dit_step.onnx   : DiT 单步前向（含 text embedding + MoE 门控/专家、adaLN、rope）
  2. vocos_spec.onnx : Vocos 主干+ISTFTHead，输出复数频谱 (real, imag)，ISTFT 在 Rust 侧做
  3. preprocess.onnx : (可选) wav -> log mel，torchaudio MelSpectrogram 导出，ORT 不支持时自动跳过

关键设计（与官方 CFM.sample 语义对齐）：
  - ODE 循环、CFG 组合、sway 时间步都在调用方（Rust / Python 基线）做，图只做单步。
  - CFG 的 null 分支在官方代码里是 drop_text=True + drop_audio_cond=True：
      * drop_text 等价于 token 全为填充符（text+1==0），调用方传全 -1 的 text 即可；
      * drop_audio_cond 等价于 cond 全零，调用方传零 cond 即可；
      * 官方在 drop_text 时跳过 MoE，本图用 moe_flag(0/1) 做算术选择，等价实现。
  - MoE 门控官方逻辑：对 text_for_gate（参考段置 -1 的 text）做 embedding 后按句平均池化，
    gate 线性层取 top1 专家（topK=1 时 softmax 权重恒为 1），专家输出 + 残差。
    本图对 9 个专家全算后用 one-hot 选择，数值等价且对 ONNX 友好。
  - 所有与序列长度相关的量（rope 频率、text 位置编码）都从 positions 输入在图内现算，
    避免 trace 把示例序列长度烘焙进图；text 输入要求调用方预先 pad/truncate 到与 mel 等长。

用法：
  python export_models.py --ckpt ../weights/10ep_mlpEXP_9.pt --out ../models
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

# 让官方仓库的 f5_tts 子模块可 import。
# 注意：不直接 import f5_tts.model 包（其 __init__ 会拉起 trainer/dataset 等训练侧重依赖），
# 而是注册一个只带 __path__ 的包占位，再逐个导入需要的子模块。
import importlib
import types

REPO_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DiaMoE-TTS", "diamoe_tts", "src")


def _import_f5():
    pkg = types.ModuleType("f5_tts")
    pkg.__path__ = [os.path.join(REPO_SRC, "f5_tts")]
    sys.modules.setdefault("f5_tts", pkg)
    sub = types.ModuleType("f5_tts.model")
    sub.__path__ = [os.path.join(REPO_SRC, "f5_tts", "model")]
    sys.modules.setdefault("f5_tts.model", sub)
    mods = {}
    for name in (
        "f5_tts.model.utils",
        "f5_tts.model.modules",
        "f5_tts.model.moe",
        "f5_tts.model.backbones.dit",
        "f5_tts.model.cfm",
    ):
        mods[name] = importlib.import_module(name)
    return mods

MEL_DIM = 100
TEXT_DIM = 512


class FriendlyMoe(nn.Module):
    """ONNX 友好的 MoE：全专家计算 + one-hot 选择（topK=1 与官方 MoeLayer 数值等价）。

    直接挂官方 MoeLayer 的 experts / gate 子模块（权重共享，不复制）。
    """

    def __init__(self, moe_layer):
        super().__init__()
        self.experts = moe_layer.experts
        self.gate = moe_layer.gate
        self.num_experts = moe_layer.num_experts

    def forward(self, text_embed, text_embed_for_gate):
        # text_embed / text_embed_for_gate: [B, N, 512]
        pooled = text_embed_for_gate.mean(dim=1)  # [B, 512]
        logits = self.gate(pooled)  # [B, E]
        idx = torch.argmax(logits, dim=1)  # [B]
        stacked = torch.stack([expert(text_embed) for expert in self.experts], dim=0)  # [E, B, N, D]
        one_hot = F.one_hot(idx, num_classes=self.num_experts).to(text_embed.dtype)  # [B, E]
        one_hot = one_hot.permute(1, 0).view(self.num_experts, -1, 1, 1)  # [E, B, 1, 1]
        selected = (stacked * one_hot).sum(dim=0)  # [B, N, D]
        # 返回选中专家编号，供量化前后的门控一致率验证（推理时忽略即可）
        return selected + text_embed, idx  # 官方 use_residual=True


class DiTStep(nn.Module):
    """DiT 单步前向（复刻官方 DiT.forward 的推理路径，去掉 cache/drop 分支）。

    输入：
      x         f32 [1, N, 100]  当前 ODE 状态（噪声 mel）
      cond      f32 [1, N, 100]  参考 mel（生成段为 0）
      text      i64 [1, N]       token id 序列（已 pad 到 N，填充 -1）
      text_gate i64 [1, N]       门控用 text（参考段置 -1）
      time      f32 [1]          当前时间步 t
      moe_flag  f32 [1]          1=启用 MoE（条件分支），0=跳过（CFG null 分支）
      positions i64 [N]          0..N-1 位置索引
    输出：
      vel       f32 [1, N, 100]  流匹配速度场预测
    """

    def __init__(self, dit):
        super().__init__()
        self.dit = dit
        self.moe = FriendlyMoe(dit.moe)

    def _text_embed(self, dit, text, seq_len, positions):
        # 复刻 TextEmbedding.forward(drop_text=False, mask_padding=False 配置)
        te = dit.text_embed
        text = text + 1  # 0 为填充符
        text = text[:, :seq_len]
        batch, text_len = text.shape[0], text.shape[1]
        if text_len < seq_len:
            text = F.pad(text, (0, seq_len - text_len), value=0)
        emb = te.text_embed(text)  # [B, N, 512]
        # 正弦位置编码（buffer freqs_cis: [4096, 512]），按 positions 查表（上限截断与官方一致）
        pos = torch.clamp(positions, max=te.precompute_max_pos - 1)
        emb = emb + te.freqs_cis[pos].unsqueeze(0)
        # ConvNeXtV2 文本建模块（mask_padding=False，无掩码逻辑）
        emb = te.text_blocks(emb)
        return emb

    def forward(self, x, cond, text, text_gate, time, moe_flag, positions):
        dit = self.dit
        seq_len = x.shape[1]
        # 按模型权重 dtype 自适应（FP32 导出时为 float，FP16 导出时为 half），IO 始终 FP32
        dtype = next(dit.parameters()).dtype
        x = x.to(dtype)
        cond = cond.to(dtype)
        time = time.to(dtype)
        moe_flag = moe_flag.to(dtype)
        t = dit.time_embed(time)  # [1, 1024]

        text_embed = self._text_embed(dit, text, seq_len, positions)
        text_embed_gate = self._text_embed(dit, text_gate, seq_len, positions)
        moe_out, gate_idx = self.moe(text_embed, text_embed_gate)
        # moe_flag 做算术选择：1 时用 MoE 输出，0 时原样（官方 drop_text 分支跳过 MoE）
        flag = moe_flag.view(1, 1, 1)
        text_embed = flag * moe_out + (1.0 - flag) * text_embed

        x = dit.input_embed(x, cond, text_embed, drop_audio_cond=False)

        # rope：按 x-transformers RotaryEmbedding 的语义从 positions 现算频率。
        # inv_freq 长度 32（dim_head=64 的一半），freqs 复制交织成 64 维；use_xpos=False -> scale=1。
        inv_freq = dit.rotary_embed.inv_freq  # [32]
        freqs = torch.einsum("n,j->nj", positions.to(inv_freq.dtype), inv_freq)  # [N, 32]
        freqs = torch.stack((freqs, freqs), dim=-1).reshape(1, -1, 64)  # [1, N, 64]，N 由运行时 positions 决定
        rope = (freqs, 1.0)

        for block in dit.transformer_blocks:
            x = block(x, t, mask=None, rope=rope)

        x = dit.norm_out(x, t)
        return dit.proj_out(x).float(), gate_idx


def build_model(ckpt_path, vocab_file):
    """按官方 load_model 的方式构建 CFM 并加载权重（use_ema 依 checkpoint 内容自动判断）。"""
    import yaml

    mods = _import_f5()
    from f5_tts.model.moe import EXPERT_DICT, MoeLayer, SimpleGate
    from f5_tts.model.utils import get_tokenizer
    from f5_tts.model.cfm import CFM
    from f5_tts.model.backbones.dit import DiT

    cfg_path = os.path.join(REPO_SRC, "f5_tts", "configs", "diamoetts.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    arch = dict(cfg["model"]["arch"])
    moe_cfg = cfg["MoE"]

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")
    print(f"vocab_size={vocab_size}")

    num_exps = int(moe_cfg["num_exps"])
    expert_cls = EXPERT_DICT[str(moe_cfg["expert_type"]).lower()]
    experts = [expert_cls() for _ in range(num_exps)]
    moe = MoeLayer(
        experts=experts,
        gate=SimpleGate(num_exps),
        num_experts=num_exps,
        num_experts_per_tok=int(moe_cfg["moe_topK"]),
        use_residual=True,
        use_dialect_clf=False,
    )

    model = CFM(
        transformer=DiT(**arch, text_num_embeds=vocab_size, mel_dim=MEL_DIM, moe=moe),
        mel_spec_kwargs=dict(
            n_fft=1024, hop_length=256, win_length=1024,
            n_mel_channels=MEL_DIM, target_sample_rate=24000, mel_spec_type="vocos",
        ),
        odeint_kwargs=dict(method="euler"),
        vocab_char_map=vocab_char_map,
        use_moe=True,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "ema_model_state_dict" in ckpt:
        state = {
            k.replace("ema_model.", ""): v
            for k, v in ckpt["ema_model_state_dict"].items()
            if k not in ("initted", "step")
        }
        print("使用 EMA 权重")
    else:
        state = ckpt["model_state_dict"]
        print("使用普通权重")
    for key in ("mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"):
        state.pop(key, None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing 示例:", missing[:10])
    if unexpected:
        print("  unexpected 示例:", unexpected[:10])
    del ckpt, state
    model.eval()
    return model


class VocosSpec(nn.Module):
    """Vocos 主干 + ISTFTHead 的线性层，输出拼接的 (real, imag) 频谱 [1, 1026, N]。

    ISTFT 不进 ONNX 图（ONNX 无 ISTFT 算子），由调用方按 torch.istft(center=True) 语义实现。
    前 513 通道为实部，后 513 通道为虚部。
    """

    def __init__(self, vocoder):
        super().__init__()
        self.backbone = vocoder.backbone
        self.head_out = vocoder.head.out

    def forward(self, mel):
        dtype = next(self.backbone.parameters()).dtype
        x = self.backbone(mel.to(dtype))  # [B, N, 512]
        x = self.head_out(x).transpose(1, 2)  # [B, 1026, N]
        mag, phase = x.chunk(2, dim=1)
        mag = torch.clip(torch.exp(mag), max=1e2)
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        return torch.cat([real, imag], dim=1).float()


class Preprocess(nn.Module):
    """wav [1, T] -> log mel [1, 100, F]（与官方 MelSpec 的 vocos 路径数值等价）。

    torchaudio 的 STFT 走复数路径，ONNX 传统导出器不支持；这里用等价的标准算子实现：
    reflect 填充 -> DFT 基卷积（stride=hop）-> 幅度 -> mel 滤波组矩阵 -> log(clamp)。
    mel 滤波组由 torchaudio 的 melscale_fbanks 在构建期算好作为常量（htk 标度，无归一化）。
    """

    def __init__(self, n_fft=1024, hop=256, win=1024, n_mels=100, sr=24000):
        super().__init__()
        from torchaudio.functional import melscale_fbanks

        self.n_fft = n_fft
        self.hop = hop
        window = torch.hann_window(win)  # periodic，与 torchaudio 默认一致
        # DFT 基：[2*513, 1, 1024]，前半实部后半虚部
        n = torch.arange(win, dtype=torch.float32)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float32)
        ang = 2 * torch.pi * torch.outer(k, n) / n_fft  # [513, 1024]
        real = torch.cos(ang) * window
        imag = -torch.sin(ang) * window
        basis = torch.cat([real, imag], dim=0).unsqueeze(1)  # [1026, 1, 1024]
        self.register_buffer("dft_basis", basis)
        fb = melscale_fbanks(
            n_freqs=n_fft // 2 + 1, f_min=0.0, f_max=sr / 2.0,
            n_mels=n_mels, sample_rate=sr, norm=None, mel_scale="htk",
        )  # [513, 100]
        self.register_buffer("mel_fb", fb)

    def forward(self, wav):
        # center=True 的 reflect 填充（n_fft//2 两侧）
        x = F.pad(wav, (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        spec = F.conv1d(x.unsqueeze(1), self.dft_basis, stride=self.hop)  # [B, 1026, F]
        n_bins = self.n_fft // 2 + 1
        real, imag = spec[:, :n_bins], spec[:, n_bins:]
        mag = torch.sqrt(real * real + imag * imag)  # power=1 即幅度谱
        mel = torch.matmul(self.mel_fb.t(), mag)  # [B, 100, F]
        return torch.log(torch.clamp(mel, min=1e-5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "DiaMoE-TTS", "diamoe_tts", "data", "vocab.txt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=256, help="导出示例序列长度（动态轴，运行时可变）")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--vocos-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        "..", "weights", "vocos"),
                    help="含 config.yaml / pytorch_model.bin 的本地 Vocos 目录")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    model = build_model(args.ckpt, args.vocab)
    dit = model.transformer

    # ---------------- DiT 单步 ----------------
    step = DiTStep(dit).eval()
    n = args.seq_len
    dummy = (
        torch.randn(1, n, MEL_DIM),
        torch.randn(1, n, MEL_DIM),
        torch.randint(0, 100, (1, n), dtype=torch.int64),
        torch.randint(0, 100, (1, n), dtype=torch.int64),
        torch.tensor([0.5]),
        torch.tensor([1.0]),
        torch.arange(n, dtype=torch.int64),
    )
    with torch.no_grad():
        ref, ref_gate = step(*dummy)
    print("DiT step 输出:", tuple(ref.shape), "gate_idx:", ref_gate.tolist())

    dit_path = os.path.join(args.out, "dit_step.onnx")
    with torch.no_grad():
        torch.onnx.export(
            step, dummy, dit_path,
            input_names=["x", "cond", "text", "text_gate", "time", "moe_flag", "positions"],
            output_names=["vel", "gate_idx"],
            dynamic_axes={
                "x": {1: "N"}, "cond": {1: "N"}, "text": {1: "N"}, "text_gate": {1: "N"},
                "positions": {0: "N"}, "vel": {1: "N"},
            },
            opset_version=args.opset,
            dynamo=False,
            do_constant_folding=True,
        )
    print("已导出", dit_path, os.path.getsize(dit_path))

    # 保存一份 PyTorch 参考输出供数值对齐
    torch.save({"inputs": [d.numpy() for d in dummy], "output": ref.numpy(),
                "gate_idx": ref_gate.numpy()},
               os.path.join(args.out, "dit_step_ref.pt"))

    # ---------------- Vocos (mel -> 频谱) ----------------
    from vocos import Vocos

    vocoder = Vocos.from_hparams(os.path.join(args.vocos_dir, "config.yaml"))
    vsd = torch.load(os.path.join(args.vocos_dir, "pytorch_model.bin"),
                     map_location="cpu", weights_only=True)
    vocoder.load_state_dict(vsd)
    vocoder.eval()
    del vsd

    spec_model = VocosSpec(vocoder).eval()
    mel_dummy = torch.randn(1, MEL_DIM, 64)
    with torch.no_grad():
        spec_ref = spec_model(mel_dummy)
    print("Vocos 频谱输出:", tuple(spec_ref.shape))
    vocos_path = os.path.join(args.out, "vocos_spec.onnx")
    with torch.no_grad():
        torch.onnx.export(
            spec_model, (mel_dummy,), vocos_path,
            input_names=["mel"], output_names=["spec"],
            dynamic_axes={"mel": {2: "N"}, "spec": {2: "N"}},
            opset_version=args.opset, dynamo=False, do_constant_folding=True,
        )
    print("已导出", vocos_path, os.path.getsize(vocos_path))
    torch.save({"input": mel_dummy.numpy(), "output": spec_ref.numpy()},
               os.path.join(args.out, "vocos_spec_ref.pt"))

    # ---------------- Preprocess (wav -> mel，可选) ----------------
    try:
        pre = Preprocess().eval()
        wav_dummy = torch.randn(1, 24000)
        with torch.no_grad():
            mel_ref = pre(wav_dummy)
        print("Preprocess mel 输出:", tuple(mel_ref.shape))
        pre_path = os.path.join(args.out, "preprocess.onnx")
        with torch.no_grad():
            torch.onnx.export(
                pre, (wav_dummy,), pre_path,
                input_names=["wav"], output_names=["mel"],
                dynamic_axes={"wav": {1: "T"}, "mel": {2: "F"}},
                opset_version=args.opset, dynamo=False, do_constant_folding=True,
            )
        print("已导出", pre_path, os.path.getsize(pre_path))
        torch.save({"input": wav_dummy.numpy(), "output": mel_ref.numpy()},
                   os.path.join(args.out, "preprocess_ref.pt"))
    except Exception as e:  # STFT 导出/ORT 支持不确定，失败不阻塞主线
        print(f"[警告] preprocess.onnx 导出失败（将走 Rust 侧 mel 实现）：{type(e).__name__}: {e}")

    print("导出完成")


if __name__ == "__main__":
    main()
