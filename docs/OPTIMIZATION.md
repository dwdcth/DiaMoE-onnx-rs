# DiaMoE-TTS 量化/推理优化方向

本文原记录"无保护 per-channel INT8 定稿"之后的优化路线。已完成实测的条目：
(b) AdaLN FP16 折中、(d) 的 A 组保护与按层敏感度扫描（上一轮），(a)
weight-only 路线与 INT4 证伪（本轮，见下）。每条写清做法、预期收益、
代价与工作量，方便按预算取舍。基线与选档的实测数字见 README 第 2 节；
本文的数字除注明"实测"外均为估算。

背景账先算清（全模型 346.6M 参数）：

- DiT 主干的大头是每层的注意力 QKV/投影与 FFN，以及 **AdaLN 调制 Linear
  （attn_norm，1024→6144）**：22 层 × 6.29M ≈ **138.4M 参数，占全模型约 40%**。
- A 组小层（moe/gate、time_embed、norm_out、proj_out）合计约 7M 参数，
  保 FP32 实测只多 10.5MB（此前估算 20–26MB，偏高），是第一批保护。
- AdaLN 若保 FP32：体积 +约 415MB（全保护图实测 819.1MB vs 无保护
  394.1MB）；保 FP16 实测 +约 137MB（531.5MB）。
- 敏感度扫描实测（见 (d)）：**单步误差大头在 FFN（block:ff，92.3M 参数，
  Δmean +0.01674，约占基线 0.06747 的 1/4）**，不在 AdaLN（+0.00276）。

---

## (a) Weight-only INT8 ——【已实测，路线修正后大胜，定稿 wo8ff / wo8full 两档】

**原设想**（W8A16：权重 INT8 + 激活全程 FP32）**在 CPU 上不成立**，但沿
MatMulNBits 路线换了个打法落地，实测全面刷新 CPU 选档：

**1. 纯 weight-only（accuracy_level 不设/unpacked 路径）被证伪：**
- 速度：wo8_v0（全图 bits=8 无 acc）单步 620ms，比 protectA 慢 +224%——
  CPU 的 unpacked 路径是"反量化成 FP32 + 普通 SGEMM"，计算量与 FP32 图
  相同还多付反量化开销（M>1 的 DiT 场景没有 M=1 解码的 GemV 专用内核）。
- 兼容性：Rust ort（ORT 1.22）对 bits=8 只注册了 acc=4 一种内核变体，
  其余直接运行时崩溃（实测 "Only 4b quantization is supported for
  unpacked compute"）。
- 误差倒是真的好（单步 mean_abs 0.0290，-57%），但速度否决。

**2. 有效路线：MatMulNBits(bits=8, block_size=128, 非对称, accuracy_level=4)。**
acc=4 的语义是"激活按相同 block_size 内部量化成 int8、int32 累加"——即
**W8 权重 + 分块 int8 激活**，不是纯 weight-only。误差改善的本质是把激活
量化粒度从 DynamicQuantizeLinear 的 **per-tensor** 提到 **per-128-block**，
权重也从 per-channel 细化到 per-block；计算走 MLAS int8 GEMM（AVX512/VNNI
快路径），与 MatMulInteger 同级。ORT 1.22 的 x86 内核对 bits=8 恰好只有
这一种变体——**现有 Rust 二进制直接可用，无需升级依赖**。

实测（8845HS，Python ORT 1.31 + Rust ort rc.10 双侧验证）：

| 图 | 构成 | 体积 | 单步 mean_abs | 单步速度 | e2e mel L1 ①/② |
|---|---|---|---|---|---|
| protectA（旧默认） | 全 int8 动态 + A⁺ | 404.6MB | 0.06531 | 基准 | 0.197/0.376 |
| **wo8ff（新默认）** | FFN→nbits8/acc4，其余 int8 动态 + A⁺ | 410.0MB | 0.03881 | -6%~+7%（≈同速） | 0.105/0.225 |
| **wo8full（新精度档）** | 全部常量 MatMul→nbits8/acc4 + Gemm int8 动态 + A⁺ | 413.2MB | 0.03182 | +16%（Rust 实测 +21%） | **0.070/0.135** |
| combo（旧上限） | FFN+AdaLN→FP16 + A⁺ | 636.1MB | 0.02504 | +90% | 0.080/0.229 |

- **wo8full 的 e2e 双用例全面超过 combo 与 fullprotect**，速度快 ~1.6 倍、
  体积少 223MB；门控 8/8（选择与 FP32 完全一致）；Rust 冒烟通过。
- **wo8ff 与 protectA 同速**（Python -6% / Rust +7%，噪声级差异），e2e 误差
  约减半——精度档 protectAp_ff16（+86% 慢）被完全支配。
- 生成脚本：`tmp_on_disk/wo_experiments.py`（build --preset matrix；三阶段
  = FP32 图上 nbits 量化指定节点 → 其余 quantize_dynamic → A⁺ 手术）。
  待办：把 build 逻辑并入 `export/quantize.py` 正式化。

**3. INT4（bits=4）证伪**：wo4full 单步 mean_abs 0.12924（比无保护 int8 还
差 +91.5%），wo4ff 0.09274（+37.5%）——int4-g128 的 RTN 权重噪声（元素级
~10%）在 F5 类流匹配 DiT 上不可接受，与 (d) 的风险预判一致。wo4 系两张图
仅留档（`dit_step_wo4_protectAp.onnx` / `dit_step_int8_ffwo4_protectAp.onnx`）。

**4. 未走的支线（记录备查）**：block_size=32（微软 Phi 系列 int4 CPU 模型
与 ORT 1.31 吞吐模式的调优甜点）可能比 g128 更快/更准，未实测；静态校准
量化 (c) 的误差上限已低于 wo8 路线（仍保留 per-tensor 激活量化），降级为
仅在需要 protectA 极速档下压误差时考虑。

## (b) AdaLN 保护的折中：FP16 而非 FP32 ——【已完成，实测结论；本轮起被 wo8 路线取代】

**实测（本轮，8845HS）：单独做收益远低于预期。** adaln16 单步 mean_abs 仅
从 0.06747 降到 0.06471（-4%），门控仍 7/8，e2e 无独立优势（README 第 2 节
端到端表）。原因由扫描揭示：attn_norm 的单步敏感度只有 +0.00276
（+138.5MB），远小于 FFN 的 +0.01674（+92MB）。

**但作为组合成分有效**：combo（FFN→FP16 + attn_norm→FP16 + A⁺→FP32，
636.1MB）单步 mean_abs 0.02504（-63%）、门控 8/8、e2e 双用例 mel/波形均
最优，全面超过全 FP32 保护的上限对照（819.1MB）。attn_norm 的价值在 e2e
而非单步：protectA_adaln16 单步仅略优于 protectA，e2e 却显著更好——单步
固定随机输入会低估 AdaLN 保护的端到端收益（ODE 逐步放大的调制误差），
这是本轮最重要的方法论结论。

体积账实测：adaln16 531.5MB / protectA_adaln16 542.0MB / combo 636.1MB
（此前估算 530–540MB 基本吻合）。

实现沉淀在 `export/quantize.py` 的 `restore_sensitive(fp16=True)`：
Cast(输入→F16) → MatMul/Gemm(F16 权重) → Cast(输出→F32)，权重以 `_fp16`
新名落图，Gemm 的 bias 同转 F16；手术含脚手架吸收红线与节点级复查。

## (c) 静态校准量化（真实方言校准集）

**做法**：不用动态量化，改静态量化：准备校准集——用官方 prompts 的多方言
参考音频 + 文本，跑 FP32 图收集每一步 ODE 中间激活的分布（注意：ODE 每步
的激活分布随 t 变化，校准样本要覆盖多个 t 与 CFG 双分支），用
Entropy/Percentile 校准法给激活定 scale，再做 per-channel 权重 + 静态激活
的 INT8 图（QDQ 格式）。

**预期收益**：静态量化在 CPU 上通常比动态量化快（省掉运行时算 scale），
且校准得当的激活误差小于动态量化的逐张量 min/max。对方言场景还有一个
额外好处：校准集覆盖各方言时，门控与专家路径的激活范围更真实。

**代价**：校准集构建与采集流水线要写（跑几十条样本 × 多个 t）；静态图对
输入分布更敏感，超出校准范围的长句/新方言可能误差放大；ORT 静态量化图
在 Rust ort 预编译二进制里的算子覆盖（QuantizeLinear/DequantizeLinear/
MatMulInteger 组合）要重新验证。

**工作量**：中到大，2–4 天（含校准集、采集、量化、多方言误差回归）。

## (d) 进一步压误差的其他选项

- **A 组小层保护**——【已完成，实测】protectA（404.6MB，+10.5MB）：
  门控 7/8 → **8/8**（修复唯一方言改选），NFE16 耗时不变，已定为 CPU 默认
  推荐档。注意 A 组必须保 FP32 而非 FP16：扫描实测 norm_out / moe 还原成
  FP16 反而劣化（Δmean -0.0041 / -0.0007）。
- **按层敏感度扫描**——【已完成，实测】`export/sensitivity_scan.py`：从
  FP32 图实际发现分组（22 整层 + 3 跨层组件 + 6 组外，共 31 组，全部跑通），
  每组在 INT8 副本上还原 FP16 测单步误差收益，结果存
  `models/sensitivity_scan.json`。要点：**block:ff（92.3M 参数/+92MB）
  Δmean +0.01674 是最大误差源**；proj_out +0.1MB 买 +0.00560、
  input_embed +0.7MB 买 +0.00377（原 A 组漏掉，已纳入 A⁺）；attn_norm
  +138.5MB 仅 +0.00276；单层排序 9 > 0 > 4 > 20 > 18 > 2 ≈ 14（首尾层
  更敏感）。扫描直接催生了 protectAp_ff16 与 combo 两档（README 第 2 节）。
- **INT4（g128 分组量化）**——【已实测，证伪】：wo4full（全图 bits=4
  acc=4）单步 mean_abs 0.12924（比无保护 int8 差 +91.5%）、wo4ff 0.09274
  （+37.5%），详见 (a) 第 3 条。仅体积有意义（318–364MB），音质不可用，
  两张图留档 models/ 供对照。
- **QAT（量化感知微调）**：拿官方数据对量化误差做短程微调，理论上限最好，
  但需要训练数据、算力与防灾难性遗忘的评估，本项目只提方向，不展开。

## (e) 推理侧提速

- **NFE 降档（已实测可算）**：ODE 步数与耗时近似线性。Python 基线 FP32
  NFE 32/16/8 分别约 112.8/57.3/26.4s（本机 2 核）。NFE=16 通常是质量/
  速度拐点，NFE=12 可作快速档；建议在目标机上用真实长句听感确认后把
  默认从 32 降到 16，**直接省一半时间**，零开发成本。
- **CFG 折叠/批处理**：当前每步跑 cond/null 两次单步前向。可把双分支拼
  batch=2 一次前向（图已支持 batch 维的做法需验证 rope/positions 广播），
  或在 GPU 上用 CUDA Graph 固定形状减少调度开销。CPU 上批处理收益有限，
  GPU 上可望再省 10–30% 的调度与访存开销。工作量小到中。
- **CUDA EP + FP16（目标机 2080 Ti 22GB）**：Rust 已留 `--features cuda`。
  2080 Ti 是 Turing、无 BF16 张量核但有 FP16 张量核，FP16 图（699.6MB）
  在 GPU 上预计比本机 CPU 快一个数量级（本机无 GPU 未实测，上目标机后
  先跑 `cargo build --release --features cuda` 并核对 ORT 的 CUDA provider
  与驱动版本）。注意 FP16 图在 CPU 上不加速（实测略慢于 FP32），CPU 场景
  用 INT8 系选档（默认 protectA、精度优先 combo，见 README 第 2 节）。
- **Vocos 与 ISTFT**：只占总耗时很小一部分（单次前向），非瓶颈，不必先动。
- **时长与分段**：长文本按句切分并行多会话（ORT session 可并发 run），
  吞吐可随核数线性扩展；本机 2 核意义不大，目标机值得做。

## (f) g2pw 前端模型的 ONNX 化与 INT8 ——【本轮已落地基础】

前端唯一的神经组件 g2pw（多音字消歧，BERT-base 分类器）**官方发行本来就是
ONNX**（`RICHARD12369/DiaMoE_TTS` 仓库的 g2pw.tar.gz，441.5MB，含
g2pw.onnx + 多音字/注音表 + bert-base-chinese tokenizer），不存在"导出"问题。
本轮已完成：

- 下载解压到 `weights/g2pW/`，Python 冒烟通过（银行/行长/行走、重庆/重量、
  朝/教 等多音字全部消歧正确，FP32 单句 21–194ms）。
- **INT8 动态量化版 `g2pw_int8.onnx`（185.2MB，-58%）**：6 句多音字测试
  与 FP32 完全一致，速度快约 3 倍；算子集（MatMul/Gemm → MatMulInteger
  + DynamicQuantizeLinear）与 DiT int8 图相同，Rust ORT 1.22 已验证可用。
- 坑：`onnx_api.py` L72 写死用 `config.model_source` 加载 tokenizer（忽略
  传入参数），离线环境需把 `weights/g2pW/config.py` 的 model_source 指向
  本地 bert-base-chinese 目录（已改）。

**剩余工作（Rust 全离线前端）——【已完成，2026-10-10】**：`rust/src/frontend.rs`
移植官方普通话管线（jieba-rs + g2pw_int8.onnx + pypinyin 词典 TSV 化 +
变调/儿化 + 拼音→IPA），`--text-hanzi` 一步出声；10 句对照集 9/10 与 Python
官方管线逐 token 一致（唯一差异：jieba-rs 对「地接」分词不同 → 「地」少轻声
变调）。余项：文本正则化（数字→汉字）、方言表导出（管线已通用）。踩坑与
验证细节见 HANDOVER §4 第 9 条。

---

## 建议的执行顺序（按性价比）

（本轮已完成 weight-only 路线实测并定稿 wo8ff / wo8full 两档，见 (a)；
INT4 证伪。余下：）

1. **人耳评测**新三档试听（output/tengwanggexu_wo8ff/wo8full/protectA.wav）
   + 用正确括号格式重跑一组 e2e 基准（现有 e2e 表仍是旧格式，仅相对有效）。
2. 目标机上 FP16 + CUDA 实测并定 NFE 默认档（未做，等 train_dino 训练结束；
   零/低成本，收益最大——GPU 预计比 CPU 快一个数量级）
3. g2pw Rust 移植（(f)）：tokenizer + 输入拼装 + jieba/pypinyin 数据化，
   补齐 Rust 全离线管线的最后一块。
4. wo8 档生成脚本并入 export/quantize.py 正式化；可选 block_size=32 调优。
5. 静态校准量化 (c)：降级为可选项（误差上限已不及 wo8 路线）；INT4/QAT
   已证伪/不展开。
