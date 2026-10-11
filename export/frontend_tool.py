#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""前端数据导出 + Python 参考管线（Rust 前端移植的对照基准）。

子命令：
  dump  把 Rust 前端需要的全部数据表导出到 frontend_data/（TSV/纯文本）：
        chars.tsv / phrases.tsv / lexicon.tsv / polyphonic.tsv /
        must_neural.tsv / must_not_neural.tsv / must_erhua.tsv / not_erhua.tsv /
        syllable_ipa.tsv / tone_mark.tsv / word_dialect.tsv / t2s.tsv
  ref   对给定句子跑上游完整管线（zh_frontend g2pW → word2dialect →
        liandu_tone → pinyin2ipa → ipa_tone），输出最终无括号 IPA，
        作为 Rust 移植的 ground truth。

用法（在仓库根目录）：
  .venv/bin/python export/frontend_tool.py dump
  .venv/bin/python export/frontend_tool.py ref sentences.txt --out ref_ipa.txt
"""

import argparse
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DF_DIR = os.path.join(ROOT, "export", "DiaMoE-TTS", "dialect_frontend")
G2PW_DIR = os.path.join(ROOT, "weights", "g2pW")
OUT_DIR = os.path.join(ROOT, "frontend_data")

sys.path.insert(0, DF_DIR)

# zh_frontend 顶层 import g2pM 且 __init__ 里就实例化（仅作 g2pW 失败的 fallback）。
# pip 装的 g2pM 需要联网拉数据，这里换成不联网的宽松 stub：调用时直接报错。
_g2pm = types.ModuleType("g2pM")


class _G2pM:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        raise RuntimeError("g2pM fallback 被调用（不应发生：g2pW 应覆盖全部输入）")


_g2pm.G2pM = _G2pM
sys.modules.setdefault("g2pM", _g2pm)


def _stub_g2pw(model_dir):
    """让 zh_frontend.Frontend 用本地 g2pw 路径实例化转换器。"""
    import frontend.zh_frontend as zf
    from frontend.g2pw.onnx_api import G2PWOnnxConverter as _Orig

    def _ctor(*a, **kw):
        kw["model_dir"] = model_dir
        return _Orig(*a, **kw)

    zf.G2PWOnnxConverter = _ctor


def _load_pinyin_state():
    """复刻 zh_frontend g2pW 分支的 pypinyin 词典状态（large + 自定义词 + 地）。"""
    from pypinyin_dict.phrase_pinyin_data import large_pinyin
    large_pinyin.load()
    from frontend import zh_frontend as zf
    import inspect
    src = inspect.getsource(zf.Frontend.__init__)
    # phrases_dict / 地 覆写直接通过实例化 Frontend(g2p_model="pypinyin") 触发
    from pypinyin import load_single_dict
    load_single_dict({ord("地"): "de,di4"})
    return src


def do_dump():
    import pandas as pd
    import tools  # noqa: F401  先触发 tools→dialect_frontend→zh_frontend 的完整加载（避免循环导入）
    from pypinyin import pinyin, Style
    from pypinyin.core import PINYIN_DICT, PHRASES_DICT

    os.makedirs(OUT_DIR, exist_ok=True)

    # 先实例化 pypinyin 分支的 Frontend 触发词典加载（large_pinyin + phrases + 地）
    from frontend.zh_frontend import Frontend
    f = Frontend(g2p_model="pypinyin")
    # 再取加载后的全局词典（Frontend 的 load 已写入 pypinyin 全局）
    from pypinyin.core import PINYIN_DICT, PHRASES_DICT  # noqa: F811 重新取

    def w(name, lines):
        p = os.path.join(OUT_DIR, name)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"{name}: {len(lines)} 行")

    # 1. chars.tsv：字 → 全部读音（TONE3 + 轻声 5 + ü→v），竖线分隔
    lines = []
    for code, readings in sorted(PINYIN_DICT.items()):
        ch = chr(code)
        try:
            variants = pinyin(ch, heteronym=True, neutral_tone_with_five=True,
                              style=Style.TONE3)[0]
        except Exception:
            variants = [readings.split(",")[0]]
        vs = [v for v in variants if v]
        if vs:
            lines.append(f"{ch}\t" + "|".join(vs))
    w("chars.tsv", lines)

    # 2. phrases.tsv：词 → 音节（lazy_pinyin 走完整管线，含风格转换）
    from pypinyin import lazy_pinyin
    lines = []
    for phr in PHRASES_DICT:
        try:
            pys = lazy_pinyin(phr, neutral_tone_with_five=True, style=Style.TONE3)
        except Exception:
            continue
        if all(pys):
            lines.append(f"{phr}\t{' '.join(pys)}")
    w("phrases.tsv", lines)

    # 3. lexicon.tsv：音节 → "声母 韵母"（含调数字在韵母尾；来自 generate_lexicon）
    from frontend.generate_lexicon import generate_lexicon
    lexicon = generate_lexicon(with_tone=True, with_erhua=True)
    lines = [f"{k}\t{v}" for k, v in sorted(lexicon.items())]
    w("lexicon.tsv", lines)

    # 4. polyphonic.tsv（词级读音纠正，来自 polyphonic.yaml）
    from frontend.polyphonic import Polyphonic
    pc = Polyphonic()
    lines = [f"{k}\t{' '.join(v) if isinstance(v, list) else v}"
             for k, v in sorted(pc.polyphonic_words.items())]
    w("polyphonic.tsv", lines)

    # 5. 声调轻声词表 / 儿化集合
    from frontend.tone_sandhi import ToneSandhi
    ts = ToneSandhi()
    w("must_neural.tsv", sorted(ts.must_neural_tone_words))
    w("must_not_neural.tsv", sorted(ts.must_not_neural_tone_words))
    w("must_erhua.tsv", sorted(f.must_erhua))
    w("not_erhua.tsv", sorted(f.not_erhua))

    # 6. 拼音→IPA 与声调符号表（putonghua 的 base_*.xlsx）
    dbase = os.path.join(DF_DIR, "frontend", "dialect", "putonghua")
    df = pd.read_excel(os.path.join(dbase, "base_syllable.xlsx"))
    lines = []
    for _, r in df.iterrows():
        py = str(r["pinyin"]).strip()
        ini = "" if pd.isna(r["initial"]) else str(r["initial"]).strip()
        fin = "" if pd.isna(r["final"]) else str(r["final"]).strip()
        lines.append(f"{py}\t{ini},{fin}")
    w("syllable_ipa.tsv", lines)

    df = pd.read_excel(os.path.join(dbase, "base_tone.xlsx"))
    w("tone_mark.tsv", [f"{str(r['contour']).strip()}\t{str(r['mark']).strip()}"
                        for _, r in df.iterrows()])

    # 7. word_dialect.tsv（word2dialect 步骤，blocked 过滤；word/dialect_pinyin）
    df = pd.read_excel(os.path.join(dbase, "base_word.xlsx"))
    lines = []
    for _, r in df.iterrows():
        if str(r.get("blocked", "")).strip().lower() == "true":
            continue
        word = str(r["word"]).strip()
        dp = str(r["dialect_pinyin"]).strip()
        lines.append(f"{word}\t{dp}")
    w("word_dialect.tsv", lines)

    # 8. t2s.tsv（繁→简映射，g2pw 输入预处理用）
    from frontend.zh_normalization.char_convert import t2s_dict
    w("t2s.tsv", [f"{k}\t{v}" for k, v in sorted(t2s_dict.items())])


def do_ref(sent_file, out_file):
    """对每行句子跑上游完整管线，输出 最终 IPA（无括号）。"""
    import tools  # noqa: F401  循环导入顺序：必须先于 zh_frontend
    _stub_g2pw(G2PW_DIR)
    from frontend.zh_frontend import Frontend
    from frontend.symbols import zh_phonemes
    f = Frontend(g2p_model="g2pW", vocab_phones=zh_phonemes)

    # 步骤 3-6 的函数直接 import（与 single_frontend.sh 同一份实现）
    sys.path.insert(0, DF_DIR)
    import importlib
    w2d = importlib.import_module("word2dialect_pinyin")
    ldt = importlib.import_module("liandu_tone")
    p2i = importlib.import_module("pinyin2ipa")
    itone = importlib.import_module("ipa_tone")
    import pandas as pd
    import re

    dbase = os.path.join(DF_DIR, "frontend", "dialect", "putonghua")
    p2i_dict = p2i.load_pinyin_to_ipa_dict(os.path.join(dbase, "base_syllable.xlsx"))
    tone_map = itone.load_tone_mapping(os.path.join(dbase, "base_tone.xlsx"))
    df = pd.read_excel(os.path.join(dbase, "base_word.xlsx"))
    text_map, hash_only_map, _ = w2d.load_word_dicts_v4(os.path.join(dbase, "base_word.xlsx"))

    sents = [ln.strip() for ln in open(sent_file, encoding="utf-8")
             if ln.strip() and not ln.startswith("#")]

    # gen_ppinyin_oop.py 的标点回转（jieba 会把 CJK 标点转成 ASCII，需转回）
    en_to_zh = {",": "，", ".": "。", "?": "？", "!": "！", ":": "：", ";": "；",
                "(": "（", ")": "）", "[": "【", "]": "】", "{": "｛", "}": "｝",
                "<": "《", ">": "》", '"': "“", "'": "‘", "-": "－", "_": "＿",
                "&": "＆", "@": "＠", "/": "／", "\\": "、", "|": "｜",
                "`": "｀", "~": "～", "^": "＾"}

    def punct_back(s):
        for en, zh in en_to_zh.items():
            s = s.replace(en, zh)
        return s

    results = []
    for idx, sent in enumerate(sents):
        phones, word2ph, tones, ppinyins, oop, zhongwens = \
            f.get_splited_phonemes_tones([sent])
        zhongwens = punct_back(zhongwens)
        ppinyins = punct_back(ppinyins)
        # 步骤 3：word2dialect（#标记 + 词级替换）
        text_with_hash, pinyin_with_hash = w2d.insert_hash_marks_aligned(
            zhongwens, ppinyins, hash_only_map)
        text_units = w2d.tokenize_text(text_with_hash)
        pinyin_units = w2d.tokenize_pinyin(pinyin_with_hash)
        replacements = sorted(text_map.items(),
                              key=lambda kv: -len(kv[0].replace("#", "")))
        pinyin_units = w2d.apply_replacements(text_units, pinyin_units, replacements)
        # 步骤 4：liandu_tone（普通话 (3,3)→(2,3) + 一/不）
        pinyin_units = ldt.apply_all_tone_changes_blocked(
            pinyin_units, text_units, ldt.TONE_RULES["putonghua"],
            ldt.CHAR_TONE_RULES["putonghua"], "putonghua")
        # 步骤 5：拼音→IPA
        text_clean = p2i.replace_english_punctuation(
            text_with_hash).replace("#", "")
        unmatched = set()
        ipa_units = p2i.convert_pinyin_to_ipa(pinyin_units, p2i_dict, unmatched)
        ipa = p2i.format_ipa_output(ipa_units)
        # 步骤 6：声调符号
        ipa = itone.replace_tone_marks_in_ipa(ipa, tone_map)
        results.append(ipa)
        if unmatched:
            print(f"[warn] 行{idx+1} 未映射拼音: {sorted(unmatched)}", file=sys.stderr)

    with open(out_file, "w", encoding="utf-8") as fh:
        fh.write("\n".join(results) + "\n")
    print(f"参考 IPA 已写入 {out_file}（{len(results)} 句）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "ref"])
    ap.add_argument("sent_file", nargs="?", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.cmd == "dump":
        do_dump()
    else:
        do_ref(args.sent_file, args.out or "ref_ipa.txt")


if __name__ == "__main__":
    main()
