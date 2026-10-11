#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""前端 IPA 串 → DiaMoE 模型输入格式（括号 token 串）。

背景（本轮合成实测时破解，详见 README 第 4 节）：DiaMoE 的模型输入格式是
**括号 token**——官方训练路径 ipa_to_tensor（utils.py）按空白切分后整 token
查词表、token 间补空格 id，而词表 vocab.txt 的音节单元都是 `[k]`、`[ˈyᴴᴸ]`
这种带方括号的行。官方 prompts/*.txt 第三列与官方推理入口 list_str_to_idx
（逐字符）都不产生这个格式（上游 train/infer 不一致）。前端输出的无括号 IPA
直接喂模型会把大部分音节 token 编码成 0，必须先经本脚本括号化。

转换规则（逐 token，按空白切分）：
  1. `[tok]` 在词表 → 输出 `[tok]`（括号化）
  2. `tok` 裸形式在词表 → 保持裸（`，。！？,.!?`、数字等标点/符号走这条）
  3. `tok == "|"` → 保持裸（`|` 不在词表，编码为 0，等价空格/音节分隔）
  4. 两者都查不到 → 报错列出全部未知 token，非零退出，不写输出

用法：
  python bracket_ipa.py <in.txt|-> [--col 3] [--out out.txt] [--vocab path]
  python bracket_ipa.py "ɥ ˈyᴴᴸ | t͡ʂ ˈɑᴴᴴ ŋ | ，"     # 直接传 IPA 串

  <in.txt>  TSV 文件取第 --col 列（1-based，默认 3，即官方 prompts 格式的
            IPA 列）；行内列数不足时回退用整行。`-` 读 stdin。
  直接传串  参数不是存在的文件且非 `-` 时按 IPA 串处理（整串转换）。
  --out     省略则打印到 stdout（多行输入逐行对应）。
  --vocab   默认 DiaMoE-TTS/diamoe_tts/data/vocab.txt。
"""

import argparse
import os
import sys


def load_vocab(path):
    vocab = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = line.rstrip("\n").strip()
            if t:
                vocab.add(t)
    return vocab


def convert(text, vocab):
    """转换单行/单个 IPA 串。返回 (输出串, 统计 dict, 未知 token 列表)。"""
    bracketed = bare = 0
    unknown = []
    out = []
    for tok in text.split():
        if f"[{tok}]" in vocab:
            out.append(f"[{tok}]")
            bracketed += 1
        elif tok in vocab:
            out.append(tok)  # 标点/数字等裸形式在词表，保持裸
            bare += 1
        elif tok == "|":
            out.append(tok)  # 音节分隔，不在词表，编码为 0（等价空格）
            bare += 1
        else:
            unknown.append(tok)
    stats = {"tokens": len(text.split()), "bracketed": bracketed, "bare": bare}
    return " ".join(out), stats, unknown


def main():
    ap = argparse.ArgumentParser(description="前端 IPA 串 → 括号 token 串（DiaMoE 模型输入格式）")
    ap.add_argument("src", help="TSV 文件路径（取第 --col 列）/ '-'（stdin）/ 直接传 IPA 串")
    ap.add_argument("--col", type=int, default=3, help="TSV 取第几列（1-based，默认 3）")
    ap.add_argument("--out", default=None, help="输出文件路径，默认打印到 stdout")
    ap.add_argument("--vocab", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "DiaMoE-TTS", "diamoe_tts", "data", "vocab.txt"))
    args = ap.parse_args()

    if not os.path.exists(args.vocab):
        raise SystemExit(f"词表不存在: {args.vocab}")
    vocab = load_vocab(args.vocab)

    if args.src == "-" or os.path.exists(args.src):
        # 文件/stdin 模式：TSV 取第 col 列，列数不足回退整行
        lines = sys.stdin.read().splitlines() if args.src == "-" \
            else open(args.src, encoding="utf-8").read().splitlines()
        src_lines = []
        for ln in lines:
            if not ln.strip():
                continue
            parts = ln.split("\t")
            src_lines.append(parts[args.col - 1].strip() if len(parts) >= args.col else ln.strip())
        from_file = True
    else:
        src_lines = [args.src.strip()]  # 直接传 IPA 串
        from_file = False

    results, unknowns, tot = [], [], {"tokens": 0, "bracketed": 0, "bare": 0}
    for s in src_lines:
        out, stats, unk = convert(s, vocab)
        results.append(out)
        unknowns.extend(unk)
        for k in tot:
            tot[k] += stats[k]
    if unknowns:
        uniq = sorted(set(unknowns))
        print(f"[错误] {len(unknowns)} 个未知 token（[t] 与裸 t 都不在词表）：{' '.join(uniq[:30])}"
              f"{' ...' if len(uniq) > 30 else ''}", file=sys.stderr)
        print("不写输出；请先修正前端输出或确认词表路径。", file=sys.stderr)
        sys.exit(1)

    text = "\n".join(results) + ("\n" if results else "")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)

    src_desc = f"{args.src} 第 {args.col} 列，{len(src_lines)} 行" if from_file else "IPA 串参数"
    print(f"bracket_ipa: 来源={src_desc}，token {tot['tokens']} 个，括号化 {tot['bracketed']}，"
          f"裸（含 |）{tot['bare']}", file=sys.stderr)


if __name__ == "__main__":
    main()
