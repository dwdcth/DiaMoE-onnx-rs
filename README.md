# DiaMoE-onnx-rs

**DiaMoE-TTS 的纯 Rust 本地推理**：汉字文本 → 语音，一条命令、单二进制、推理时零 Python 依赖。
含 CPU 量化模型选档（实测 e2e 误差优于官方 FP32 保护组合）、内置 g2pw 文本前端、
预置音色与自动择优（best-of）。

- 上游模型：[GiantAILab/DiaMoE-TTS](https://github.com/GiantAILab/DiaMoE-TTS)（F5 架构多方言 MoE TTS，代码 MIT，**权重 CC-BY-NC-4.0 非商用**）
- 运行时：[ONNX Runtime](https://onnxruntime.ai/)（ort crate 静态链接）+ [jieba-rs](https://github.com/messense/jieba-rs) + [HuggingFace tokenizers](https://github.com/huggingface/tokenizers)
- 声码器：[Vocos](https://github.com/gemelo-ai/vocos)（ISTFT 在 Rust 侧实现）

## 功能

| 功能 | 说明 |
|---|---|
| 全离线合成 | `--text-hanzi "汉字文本"` 直接出 wav：分词 → g2pw 多音字消歧（ONNX）→ 变调/儿化 → IPA → 流匹配 ODE → Vocos |
| 预置音色 | `--voice chengdu_female` 等 18 个官方参考音色（EdgeTTS 式体验，参考文本自动配对） |
| CPU 量化档 | wo8ff（默认）/ wo8full（精度）/ int8_gate（最小）/ fp16（GPU），见下方精度表 |
| 自动择优 | `--best-of N`：每块并发生成 N 个候选，mel medoid 自动挑（无需 ASR/人耳） |
| IPA 输入 | 也支持直接喂官方前端产出的 IPA（`--text` + `--bracket-ipa`） |
| 自带文本前端 | g2pw int8（185MB）+ pypinyin 词典数据（TSV），与官方 Python 管线 10 句对照 9/10 逐 token 一致 |

## 快速开始

### 1. 下载模型（[Releases](../../releases)）

```
models/                     # 放到仓库根目录
├── dit_step_int8_ffwo8_protectAp.onnx   # 默认档（必选其一）
├── dit_step_wo8_protectAp.onnx          # 精度档（可选）
├── dit_step_int8_gate.onnx              # 最小档（可选）
├── dit_step_fp16.onnx                   # GPU 档（可选）
├── vocos_spec.onnx / vocos_spec_fp16.onnx
├── preprocess.onnx
└── vocab.txt
frontend_data/              # --text-hanzi 需要（frontend_data.tar.gz 解压）
g2pW/                       # --text-hanzi 需要（g2pw-int8.tar.gz 解压）
voices/                     # --voice 需要（voices.tar.gz 解压）
```

### 2. 编译

```bash
cd rust
cargo build --release            # CPU 版
# NVIDIA GPU：cargo build --release --features cuda（配 fp16 图）
```

### 3. 合成

```bash
cd rust
./target/release/diamoe-tts \
  --model-dir ../models --dit dit_step_int8_ffwo8_protectAp.onnx \
  --voice xian_female \
  --text-hanzi "各位朋友大家好，欢迎收听今天的节目。" \
  --out out.wav --nfe 32 --seed 42 --threads 8

# 自动择优（每块 3 个候选取 medoid）
./target/release/diamoe-tts ... --best-of 3

# 自备参考音频（声音克隆）
./target/release/diamoe-tts ... \
  --ref-audio ref.wav --ref-text-hanzi "参考音频说的话" --text-hanzi "要合成的文本"

# 只看前端转换结果（不加载模型）
./target/release/diamoe-tts --model-dir ../models --print-ipa --text-hanzi "你好世界"
```

参考音频建议 5~12s、干净、结尾自然收住（程序会自动做首尾静音修剪、
尾垫静音、响度归一化）。**生成语速克隆自参考**——嫌快优先换慢速音色
（xian_female 3.8 字/s、cantonese_female 3.0 字/s）而不是调 `--speed`。

## 模型精度（本机实测：AMD 8845HS，Python ORT 1.31 CPU）

| 模型 | 体积 | 单步 mean_abs¹ | 门控² | e2e mel L1 用例①/②³ | 速度 vs gate 档 |
|---|---|---|---|---|---|
| **dit_step_int8_ffwo8_protectAp**（默认） | 410 MB | 0.0388 | 8/8 | 0.105 / 0.225 | ≈同速 |
| **dit_step_wo8_protectAp**（精度档） | 413 MB | 0.0318 | 8/8 | **0.070 / 0.135** | +16~21% |
| dit_step_int8_gate（最小档） | 394 MB | 0.0675 | 8/8 | ≈无保护档 | 基准 |
| dit_step_int8（无保护，不推荐） | 394 MB | 0.0675 | 7/8 | 0.133 / 0.347 | 基准 |
| dit_step_fp16（GPU/CUDA 用） | 700 MB | 0.0049 | — | — | GPU 专用 |

¹ 单步输出 vs FP32（固定输入）；² 8 方言门控选择与 FP32 一致数；
³ 端到端合成 mel L1（NFE=16，vs FP32），越小越好。旧精度档对照：
protectA（404MB，0.065/8/8，0.197/0.376）、protectAp_ff16（499MB，
0.036，+86% 慢）、combo（636MB，0.025，0.080/0.229，+90% 慢）——
均被 wo8 系支配。INT4 与纯 W8A16 已实测证伪（误差恶化），详见
[docs/OPTIMIZATION.md](docs/OPTIMIZATION.md)。

wo8 档原理：`MatMulNBits(bits=8, block=128, accuracy_level=4)`——权重
int8 分块存储，激活按 128-block int8 量化 + int32 累加（粒度远细于动态
量化的 per-tensor），走 AVX-512/VNNI 快路径。ORT 1.22 的 Rust 预编译
内核恰好只支持这一种 8-bit 变体，开箱即用。

人耳抽查（16 字样例，三档听感无差异）；客观校验可用
`export/check_asr.py`（faster-whisper 转写）。

## 已知限制（F5 架构固有）

- **偶发音节插入/重复**：非自回归流匹配的随机伪影（同 IPA 不同 seed 结果
  不同；官方 FP32/PyTorch 同样存在，已实测对齐）。现代日常文本概率低
  （实测多数 seed 逐字正确）；文言/生僻字概率高。缓解：`--best-of 3`、
  换 `--seed`、换音色。叠词（聊聊/试试）是卡顿重灾区
- 生成语速克隆自参考音频（官方公式行为）
- 文本前端只内置普通话表（方言需自行导出表，管线已通用）；不做数字/英文
  正则化（请预先转汉字）
- 前端与官方 Python 管线 10 句对照 9/10 一致（唯一差异：jieba-rs 对
  「地接」分词不同导致「地」轻声变调缺失）

## 目录结构

```
rust/        Rust 推理程序（main.rs 管线 + frontend.rs 文本前端）
export/      模型导出/量化/验证脚本（Python，仅构建期使用）
  export_models.py    FP32 三图导出（dit_step/vocos_spec/preprocess）
  quantize.py         fp16 / int8 动态量化 + 敏感层保护手术
  frontend_tool.py    文本前端数据导出（dump）与 Python 参考管线（ref）
  check_asr.py        faster-whisper 输出客观校验
  其他                 数值对齐/门控/敏感度扫描/端到端对比
docs/        量化路线与实测记录
```

模型复现全流程见 [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) 与
`export/` 脚本；wo8 图也可由 `quantize.py int8` 产出的基础图 +
MatMulNBitsQuantizer 重建（`tmp_on_disk/wo_experiments.py` 的逻辑已在
docs 中描述）。

## 许可与致谢

- 本仓库代码：Apache-2.0（见 LICENSE）
- DiaMoE-TTS 权重与官方 prompts：**CC-BY-NC-4.0（非商用）**，来自
  [GiantAILab/DiaMoE_TTS](https://huggingface.co/GiantAILab/DiaMoE_TTS)
- g2pw 多音字模型：来自 [RICHARD12369/DiaMoE_TTS](https://huggingface.co/RICHARD12369/DiaMoE_TTS)
  （int8 量化版由本项目生成，多音字测试与 FP32 全同）
- 文本前端词典数据：pypinyin（MIT）数据导出 + jieba-rs（MIT）
- Vocos：charactr/vocos-mel-24khz

**商用前必须解决模型权重的 CC-BY-NC 许可问题。**
