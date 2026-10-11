//! diamoe-tts: DiaMoE-TTS 的 Rust 推理程序。
//!
//! 管线（与官方 CFM.sample / infer 语义对齐，图划分见 export/export_models.py）：
//!   参考 wav --(preprocess.onnx)--> log mel 条件
//!   文本 --(vocab 最长匹配 token 化)--> token ids
//!   Euler ODE 循环（Rust 侧）：每步两次调用 dit_step.onnx 做 CFG，sway 时间步
//!   生成 mel --(vocos_spec.onnx)--> 复数频谱 --(Rust 侧 ISTFT)--> wav
//!
//! 注意：输入文本应是经官方 IPA 前端处理过的 token 串（如 prompts/*.txt 的第三列），
//! 普通汉字文本需先走官方 dialect_frontend 转 IPA，见 README。

mod frontend;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use ort::session::Session;
use ort::value::Tensor;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rustfft::{num_complex::Complex, FftPlanner};

const SR: usize = 24000;
const HOP: usize = 256;
const N_FFT: usize = 1024;
const MEL_CH: usize = 100;
const TARGET_RMS: f32 = 0.1;

#[derive(Parser, Debug)]
#[command(version, about = "DiaMoE-TTS Rust 推理（ONNX Runtime）")]
struct Args {
    /// 模型目录（含 dit_step*.onnx / vocos_spec*.onnx / preprocess.onnx / vocab.txt）
    #[arg(long)]
    model_dir: PathBuf,
    /// DiT 模型文件名（相对 model-dir）或路径；默认 INT8 版
    #[arg(long, default_value = "dit_step_int8.onnx")]
    dit: String,
    /// Vocos 模型文件名（相对 model-dir）；不填则按 dit 是否 fp16 自动选
    #[arg(long)]
    vocos: Option<String>,
    /// vocab.txt 路径；默认 model-dir/vocab.txt
    #[arg(long)]
    vocab: Option<PathBuf>,
    /// 参考音频 wav（任意采样率/声道，自动转 24kHz 单声道）
    #[arg(long)]
    ref_audio: Option<PathBuf>,
    /// 参考音频对应的文本（IPA token 串）
    #[arg(long)]
    ref_text: Option<String>,
    /// 要合成的目标文本（IPA token 串）
    #[arg(long)]
    text: Option<String>,
    /// 输出 wav 路径
    #[arg(long)]
    out: Option<PathBuf>,
    /// ODE 步数（NFE），官方默认 32；16/8 更快、质量略降
    #[arg(long, default_value_t = 32)]
    nfe: usize,
    /// CFG 强度，官方默认 2.0
    #[arg(long, default_value_t = 2.0)]
    cfg: f32,
    /// sway 采样系数，官方默认 -1.0
    #[arg(long, default_value_t = -1.0, allow_negative_numbers = true)]
    sway: f32,
    /// 随机种子（噪声；与 PyTorch 的种子语义不同，仅保证本程序内可复现）
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// 语速（>1 更快），用于估算生成时长
    #[arg(long, default_value_t = 1.0)]
    speed: f32,
    /// 固定总时长（秒，含参考段）；不填则按文本长度比例估算（官方公式）
    #[arg(long)]
    fix_duration_sec: Option<f32>,
    /// 输入是无括号 IPA 串时自动括号化（export/bracket_ipa.py 的内置版）。
    /// 幂等：已括号化的 token 会被裸形式分支原样放行，括号串输入不受影响。
    #[arg(long)]
    bracket_ipa: bool,
    /// 生成侧文本直接给汉字（走内置前端：分词/g2pw 多音字/变调/IPA/括号化，
    /// 与 --text 二选一）
    #[arg(long)]
    text_hanzi: Option<String>,
    /// 参考侧文本给汉字（与 --ref-text 二选一；语义同 --text-hanzi）
    #[arg(long)]
    ref_text_hanzi: Option<String>,
    /// 前端数据目录（export/frontend_tool.py dump 的产物 frontend_data/）
    #[arg(long)]
    frontend_data: Option<PathBuf>,
    /// g2pw 模型目录（含 g2pw_int8.onnx / g2pw.onnx 与 bert-base-chinese/）
    #[arg(long)]
    g2pw_dir: Option<PathBuf>,
    /// 只打印前端转换结果（无括号 IPA 与括号化两行），不加载模型不合成
    #[arg(long)]
    print_ipa: bool,
    /// 内置音色名（代替 --ref-audio/--ref-text）：如 chengdu_female、cantonese_male。
    /// 从 --voices-dir 读 <名>_prompt.wav + <名>_prompt.txt（官方 prompts 格式，
    /// 第 3 列 IPA 作为参考文本，自动括号化）
    #[arg(long)]
    voice: Option<String>,
    /// 内置音色目录（默认 ../voices 或 ./voices）
    #[arg(long)]
    voices_dir: Option<PathBuf>,
    /// 自动择优候选数：>1 时每块生成 N 个独立 seed 的候选，在 mel 空间挑
    /// medoid（最居中者）——伪影是随机离群点，干净解彼此聚集。零 ASR/人耳。
    #[arg(long, default_value_t = 1)]
    best_of: usize,
    /// onnxruntime intra-op 线程数
    #[arg(long, default_value_t = 2)]
    threads: usize,
}

// ---------------- 文本 token 化（vocab 最长匹配，与 baseline.py 一致） ----------------

struct Tokenizer {
    map: HashMap<String, i64>,
    max_len: usize,
}

impl Tokenizer {
    fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("读取 vocab 失败: {}", path.display()))?;
        let mut map = HashMap::new();
        let mut max_len = 1;
        for (i, line) in content.split('\n').enumerate() {
            let tok = line.strip_suffix('\r').unwrap_or(line);
            map.insert(tok.to_string(), i as i64);
            max_len = max_len.max(tok.chars().count());
        }
        Ok(Self { map, max_len })
    }

    fn encode(&self, s: &str) -> Vec<i64> {
        let chars: Vec<char> = s.chars().collect();
        let mut ids = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let mut found: Option<usize> = None;
            let max_l = self.max_len.min(chars.len() - i);
            for l in (1..=max_l).rev() {
                let sub: String = chars[i..i + l].iter().collect();
                if self.map.contains_key(&sub) {
                    found = Some(l);
                    break;
                }
            }
            match found {
                Some(l) => {
                    let sub: String = chars[i..i + l].iter().collect();
                    ids.push(self.map[&sub]);
                    i += l;
                }
                None => {
                    ids.push(0); // 未知 token -> 0（官方 get(c, 0) 语义）
                    i += 1;
                }
            }
        }
        ids
    }
}

/// 前端 IPA 串 → 括号 token 串（模型训练格式，见 README 第 4 节）。
/// 逐 token：`[tok]` 在词表 → 加括号；裸形式在词表 → 保持（标点等）；
/// `|` 保持裸（编码为 0，等价音节分隔）；都查不到则报错（与 Python 版一致，
/// 不静默吞未知 token）。
fn bracket_ipa(text: &str, tok: &Tokenizer) -> Result<String> {
    let mut out = Vec::new();
    let mut unknown = Vec::new();
    for t in text.split_whitespace() {
        let br = format!("[{t}]");
        if tok.map.contains_key(&br) {
            out.push(br);
        } else if tok.map.contains_key(t) || t == "|" {
            out.push(t.to_string());
        } else {
            unknown.push(t.to_string());
        }
    }
    if !unknown.is_empty() {
        anyhow::bail!(
            "bracket-ipa: {} 个未知 token（[t] 与裸 t 都不在词表）: {}",
            unknown.len(),
            unknown.join(" ")
        );
    }
    Ok(out.join(" "))
}

// ---------------- 音频读写与重采样 ----------------

fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("打开参考音频失败: {}", path.display()))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    let ch = spec.channels as usize;
    let mono: Vec<f32> = if ch > 1 {
        samples
            .chunks(ch)
            .map(|c| c.iter().sum::<f32>() / ch as f32)
            .collect()
    } else {
        samples
    };
    Ok((mono, spec.sample_rate))
}

/// 线性插值重采样（参考音频条件提取对重采样精度不敏感；如需更高质量可先在外部转码）。
fn resample_linear(wav: &[f32], from: u32, to: usize) -> Vec<f32> {
    if from as usize == to {
        return wav.to_vec();
    }
    let out_len = (wav.len() as f64 * to as f64 / from as f64).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * from as f64 / to as f64;
        let idx = pos.floor() as usize;
        let frac = (pos - idx as f64) as f32;
        let a = wav.get(idx).copied().unwrap_or(0.0);
        let b = wav.get(idx + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }
    out
}

fn write_wav(path: &Path, wav: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SR as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in wav {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    Ok(())
}

// ---------------- ISTFT（torch.istft(center=True) 等价实现） ----------------

/// spec_re / spec_im: [513 * frames] 行主序（频点 x 帧）。输出长度 (frames-1)*HOP。
fn istft(spec_re: &[f32], spec_im: &[f32], frames: usize) -> Vec<f32> {
    let bins = N_FFT / 2 + 1;
    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(N_FFT);
    // periodic Hann 窗
    let window: Vec<f32> = (0..N_FFT)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / N_FFT as f32).cos()))
        .collect();

    let out_len = (frames - 1) * HOP + N_FFT;
    let mut y = vec![0.0f32; out_len];
    let mut env = vec![0.0f32; out_len];
    let mut buf = vec![Complex::new(0.0f32, 0.0f32); N_FFT];

    for f in 0..frames {
        // 组装共轭对称全频谱后做 IFFT（等价 irfft，结果取实部）
        for k in 0..bins {
            buf[k] = Complex::new(spec_re[k * frames + f], spec_im[k * frames + f]);
        }
        for k in bins..N_FFT {
            let m = N_FFT - k;
            buf[k] = buf[m].conj();
        }
        ifft.process(&mut buf);
        let start = f * HOP;
        for i in 0..N_FFT {
            let v = buf[i].re / N_FFT as f32 * window[i];
            y[start + i] += v;
            env[start + i] += window[i] * window[i];
        }
    }
    for i in 0..out_len {
        if env[i] > 1e-11 {
            y[i] /= env[i];
        }
    }
    // center=True 两侧各去掉 N_FFT/2
    let pad = N_FFT / 2;
    y[pad..out_len - pad].to_vec()
}

// ---------------- ONNX 会话 ----------------

fn build_session(path: &Path, threads: usize) -> Result<Session> {
    let builder = Session::builder()?
        .with_intra_threads(threads)?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?;
    #[cfg(feature = "cuda")]
    let builder = builder.with_execution_providers([ort::execution_providers::CUDAExecutionProvider::default().build()])?;
    let session = builder
        .commit_from_file(path)
        .with_context(|| format!("加载 ONNX 模型失败: {}", path.display()))?;
    Ok(session)
}

fn resolve(model_dir: &Path, name: &str) -> PathBuf {
    let p = PathBuf::from(name);
    if p.exists() {
        p
    } else {
        model_dir.join(name)
    }
}

/// 生成文本分块（官方 F5 的 chunk 策略）：先按句末标点（。！？；，）切成
/// 句元，再贪心合并成 target_tokens 大小的块（~3s 音频；太碎会重复支付参考
/// 前缀算力，太长时长估算易失准触发伪影），超长单句在 `|` 处硬切。
fn split_chunks(gen_text: &str, target_tokens: usize) -> Vec<String> {
    const SENT_END: &str = "。！？；，";
    // 1. 切句元（保留句末标点）
    let mut sentences: Vec<Vec<&str>> = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    for t in gen_text.split_whitespace() {
        cur.push(t);
        let bare = t.trim_matches(|c| c == '[' || c == ']');
        let at_end = bare.chars().count() == 1 && SENT_END.contains(bare);
        if at_end {
            while cur.last() == Some(&"|") && cur.len() > 1 {
                cur.pop();
            }
            sentences.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        while cur.last() == Some(&"|") && cur.len() > 1 {
            cur.pop();
        }
        sentences.push(cur);
    }
    // 2. 贪心合并到 target_tokens；单句超 1.5×target 时在 | 硬切
    let mut chunks: Vec<String> = Vec::new();
    let mut acc: Vec<&str> = Vec::new();
    for mut s in sentences {
        if s.len() > target_tokens * 3 / 2 {
            // 超长句硬切
            if !acc.is_empty() {
                chunks.push(acc.join(" "));
                acc = Vec::new();
            }
            let mut piece: Vec<&str> = Vec::new();
            for t in s.drain(..) {
                piece.push(t);
                if piece.len() >= target_tokens && t == "|" {
                    chunks.push(piece.join(" "));
                    piece = Vec::new();
                }
            }
            if !piece.is_empty() {
                chunks.push(piece.join(" "));
            }
            continue;
        }
        if acc.len() + s.len() > target_tokens && !acc.is_empty() {
            chunks.push(acc.join(" "));
            acc = Vec::new();
        }
        acc.extend(s.drain(..));
    }
    if !acc.is_empty() {
        chunks.push(acc.join(" "));
    }
    if chunks.is_empty() {
        vec![gen_text.to_string()]
    } else {
        chunks
    }
}

/// 统计括号 IPA 串的音节数与标点数：音节 = `|` 分隔的单元（多 token 音节
/// 算一个），裸标点（，。！？；）单列。用于按时长公式按音节折算。
fn count_syllables(bracketed: &str) -> (usize, usize) {
    const PUNCT: &str = "，。！？；,.!?;";
    let mut syl = 0usize;
    let mut punct = 0usize;
    let mut in_syl = false;
    for t in bracketed.split_whitespace() {
        if t == "|" {
            in_syl = false;
        } else {
            let bare = t.trim_matches(|c| c == '[' || c == ']');
            if bare.chars().count() == 1 && PUNCT.contains(bare) {
                punct += 1;
                in_syl = false;
            } else if !in_syl {
                syl += 1;
                in_syl = true;
            }
        }
    }
    (syl, punct)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t_start = Instant::now();

    let model_dir = args.model_dir.clone();
    let dit_path = resolve(&model_dir, &args.dit);
    let vocos_name = args.vocos.clone().unwrap_or_else(|| {
        if args.dit.contains("fp16") && model_dir.join("vocos_spec_fp16.onnx").exists() {
            "vocos_spec_fp16.onnx".to_string()
        } else {
            "vocos_spec.onnx".to_string()
        }
    });
    let vocos_path = resolve(&model_dir, &vocos_name);
    let pre_path = model_dir.join("preprocess.onnx");
    let vocab_path = args.vocab.clone().unwrap_or_else(|| model_dir.join("vocab.txt"));

    // ---- 汉字前端（--text-hanzi / --ref-text-hanzi：分词+g2pw+变调+IPA+括号化）----
    let mut hanzi_gen_ipa: Option<String> = None;
    let mut hanzi_ref_ipa: Option<String> = None;
    if args.text_hanzi.is_some() || args.ref_text_hanzi.is_some() {
        let data_dir = args.frontend_data.clone().unwrap_or_else(|| {
            let p = PathBuf::from("../frontend_data");
            if p.exists() { p } else { PathBuf::from("frontend_data") }
        });
        let g2pw_dir = args.g2pw_dir.clone().unwrap_or_else(|| {
            let p = PathBuf::from("../weights/g2pW");
            if p.exists() { p } else { PathBuf::from("weights/g2pW") }
        });
        eprintln!("加载文本前端: data={} g2pw={}", data_dir.display(), g2pw_dir.display());
        let mut fe = frontend::Frontend::load(&data_dir, &g2pw_dir)?;
        if let Some(h) = &args.text_hanzi {
            hanzi_gen_ipa = Some(fe.hanzi_to_ipa(h)?);
        }
        if let Some(h) = &args.ref_text_hanzi {
            hanzi_ref_ipa = Some(fe.hanzi_to_ipa(h)?);
        }
        // 前端输出是无括号 IPA：统一在本程序内完成括号化（幂等）
        let tok_pre = Tokenizer::load(&vocab_path)?;
        if let Some(s) = &hanzi_gen_ipa {
            hanzi_gen_ipa = Some(bracket_ipa(s, &tok_pre)?);
        }
        if let Some(s) = &hanzi_ref_ipa {
            hanzi_ref_ipa = Some(bracket_ipa(s, &tok_pre)?);
        }
        if args.print_ipa {
            if let Some(s) = &hanzi_ref_ipa {
                println!("ref: {s}");
            }
            if let Some(s) = &hanzi_gen_ipa {
                println!("gen: {s}");
            }
            return Ok(());
        }
    }

    // ---- --voice：内置参考对（官方 prompts 的音频 + IPA 文本） ----
    let mut voice_ref_ipa: Option<String> = None;
    let ref_audio_path = if let Some(v) = &args.voice {
        let vdir = args.voices_dir.clone().unwrap_or_else(|| {
            let p = PathBuf::from("../voices");
            if p.exists() { p } else { PathBuf::from("voices") }
        });
        let wav = vdir.join(format!("{v}_prompt.wav"));
        let txt = vdir.join(format!("{v}_prompt.txt"));
        if !wav.exists() || !txt.exists() {
            let mut list: Vec<String> = std::fs::read_dir(&vdir).into_iter().flatten()
                .filter_map(|e| e.ok()).map(|e| e.path())
                .filter(|p| p.extension().map(|x| x == "wav").unwrap_or(false))
                .filter_map(|p| p.file_stem().and_then(|s| s.to_str())
                    .map(|s| s.trim_end_matches("_prompt").to_string()))
                .collect();
            list.sort();
            anyhow::bail!("音色 {v} 不存在（可用：{}）", list.join(", "));
        }
        let content = std::fs::read_to_string(&txt)
            .with_context(|| format!("读取音色文本失败: {}", txt.display()))?;
        // 官方 prompts 格式：名称\t汉字\tIPA；第 3 列作为参考文本（无括号 IPA）
        let ipa = content.trim().split('\t').nth(2)
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| content.trim().to_string());
        voice_ref_ipa = Some(ipa);
        wav
    } else {
        args.ref_audio.clone()
            .context("缺少 --ref-audio（或改用 --voice <音色名>，或 --print-ipa 只做前端）")?
    };
    if args.ref_text.is_none() && args.ref_text_hanzi.is_none() && voice_ref_ipa.is_none() {
        anyhow::bail!("缺少 --ref-text / --ref-text-hanzi / --voice");
    }
    if args.text.is_none() && args.text_hanzi.is_none() {
        anyhow::bail!("缺少 --text / --text-hanzi");
    }
    let out_path = args.out.clone().context("缺少 --out")?;

    // ---- 参考音频 ----
    let (wav_raw, sr) = read_wav_mono(&ref_audio_path)?;
    let mut wav = resample_linear(&wav_raw, sr, SR);
    let rms = (wav.iter().map(|v| v * v).sum::<f32>() / wav.len().max(1) as f32).sqrt();
    let mut out_scale = 1.0f32;
    if rms > 0.0 && (rms - TARGET_RMS).abs() / TARGET_RMS > 0.05 {
        // 双向归一化到 target（官方只放大过轻参考；过响参考（如 nanjing prompt）
        // 会让输出 peak>1 在写 wav 时削波，这里对称压低）
        let g = TARGET_RMS / rms;
        wav.iter_mut().for_each(|v| *v *= g);
        out_scale = rms / TARGET_RMS;
    }
    // 参考音频首尾静音修剪 + 尾部补静音：参考最后往往贴着语音结尾（实测
    // chengdu/nanjing prompt 尾静音仅 0.04~0.08s），流匹配会把尾音直接续写进
    // 生成段（whisper 实测生成开头复述参考末句"肯定不得错"/"不要太摆噢"）。
    // 官方 remove_silence_edges + 50ms pad 的强化版：5% 峰值能量阈值修剪 +
    // 0.4s 静音垫，给模型一个干净的 语音→静音→生成 过渡。
    {
        let win = (SR as f32 * 0.02) as usize; // 20ms 窗
        if wav.len() > win * 2 {
            let env: Vec<f32> = wav.chunks(win)
                .map(|c| (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).sqrt())
                .collect();
            let thr = env.iter().cloned().fold(0.0f32, f32::max) * 0.05;
            let n = env.len();
            let mut head = 0;
            while head < n && env[head] < thr { head += 1; }
            let mut tail = n;
            while tail > head && env[tail - 1] < thr { tail -= 1; }
            if tail > head {
                let s = head * win;
                let e = (tail * win).min(wav.len());
                wav = wav[s..e].to_vec();
            }
        }
        let tail_pad = (0.6 * SR as f32) as usize;
        wav.extend(std::iter::repeat(0.0f32).take(tail_pad));
    }

    // ---- 会话 ----
    let mut pre = build_session(&pre_path, args.threads)?;
    let mut dit = build_session(&dit_path, args.threads)?;
    let mut voc = build_session(&vocos_path, args.threads)?;
    eprintln!("模型已加载: dit={} vocos={}", dit_path.display(), vocos_path.display());

    // ---- 参考 mel ----
    let wav_tensor = Tensor::from_array(([1usize, wav.len()], wav))?;
    let pre_out = pre.run(ort::inputs!["wav" => wav_tensor])?;
    let (_, mel_data) = pre_out["mel"].try_extract_tensor::<f32>()?;
    // mel_data: [1, 100, F] -> cond_mel [F][100]
    let frames = mel_data.len() / MEL_CH;
    let ref_len = frames;
    let mut cond_mel = vec![0.0f32; frames * MEL_CH];
    for f in 0..frames {
        for c in 0..MEL_CH {
            cond_mel[f * MEL_CH + c] = mel_data[c * frames + f];
        }
    }
    eprintln!("参考 mel: {ref_len} 帧 ({:.2}s)", ref_len as f32 * HOP as f32 / SR as f32);

    // ---- 文本 ----
    let tok = Tokenizer::load(&vocab_path)?;
    // 优先级：--text-hanzi（已括号化）> voice 的参考 IPA（需括号化）> --text；
    // --bracket-ipa 时对 IPA 串自动括号化（幂等，括号串原样通过）
    let (ref_text, gen_text) = {
        let mut r = hanzi_ref_ipa.clone()
            .or_else(|| voice_ref_ipa.clone())
            .unwrap_or_else(|| args.ref_text.clone().unwrap_or_default());
        let g = hanzi_gen_ipa.clone()
            .unwrap_or_else(|| args.text.clone().unwrap_or_default());
        if args.bracket_ipa || voice_ref_ipa.is_some() {
            r = bracket_ipa(&r, &tok)?;
            (r, bracket_ipa(&g, &tok)?)
        } else {
            (r, g)
        }
    };
    let ref_ids = tok.encode(&ref_text);
    let (ref_syllables, _) = count_syllables(&ref_text);
    let ref_syllables = ref_syllables.max(1);

    // ---- 文本分块（官方 F5 的伪影缓解策略，utils_infer 的 chunk 逻辑）----
    // 非自回归流匹配对长句的插入/重复伪影主要来自「时长估算与模型内在节奏
    // 失配」：按句读切成短块后每块时长几乎总是准的，伪影概率大幅下降；
    // 块间 0.1s 余弦交叉淡入淡出拼接。--fix-duration-sec 时保持单块。
    let gen_chunks: Vec<String> = if args.fix_duration_sec.is_some()
        || gen_text.split_whitespace().count() <= 80
    {
        // 短文本（≤80 token ≈ 5s）单块：实测分块对短句无收益（各块重掷采样
        // 骰子 + 重复支付参考前缀算力），长文本才需要分块缓解时长失配
        vec![gen_text.clone()]
    } else {
        split_chunks(&gen_text, 60)
    };
    eprintln!("合成 {} 块: {:?}", gen_chunks.len(),
              gen_chunks.iter().map(|c| c.split(' ').count()).collect::<Vec<_>>());


    // sway 时间步（各块共用）
    let steps = args.nfe;
    let mut ts: Vec<f32> = (0..=steps).map(|i| i as f32 / steps as f32).collect();
    for t in ts.iter_mut() {
        *t += args.sway * ((std::f32::consts::FRAC_PI_2 * *t).cos() - 1.0 + *t);
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let normal = Normal::new(0.0f32, 1.0f32).unwrap();

    // 单块合成：返回该块生成段 mel（未过 Vocos，供 --best-of 在 mel 层择优）
    let synth_chunk = |dit: &mut Session,
                       chunk_text: &str,
                       rng: &mut rand::rngs::StdRng|
     -> Result<(Vec<f32>, usize)> {
        let chunk_ids = tok.encode(chunk_text);
        // 时长按音节数估（token 数混合了 | 分隔符与多 token 音节，文言双元音
        // 音节会被系统性低估——正是时长失配触发伪影的诱因）；标点按 12 帧
        // 停顿折算（~0.13s）
        let (chunk_syl, chunk_punct) = count_syllables(chunk_text);
        let duration = if let Some(sec) = args.fix_duration_sec {
            ((sec * SR as f32 / HOP as f32) as usize).max(ref_len + 1)
        } else {
            (ref_len as f64
                + (ref_len as f64 / ref_syllables as f64)
                    * (chunk_syl as f64 + 0.9 * chunk_punct as f64)
                    / args.speed as f64) as usize
        }
        .max(ref_len + 16); // 生成段下限 ~0.17s，防极短块退化

        // full = ref + 空格 + chunk；gate 把参考段掩 -1（门控只看目标文本）
        let mut full_ids = ref_ids.clone();
        full_ids.push(0);
        full_ids.extend_from_slice(&chunk_ids);
        let mut gate_ids = vec![-1i64; ref_ids.len() + 1];
        gate_ids.extend_from_slice(&chunk_ids);
        let neg_ids = vec![-1i64; full_ids.len()];
        let mut text = vec![-1i64; duration];
        for (i, &id) in full_ids.iter().take(duration).enumerate() {
            text[i] = id;
        }
        let mut text_gate = vec![-1i64; duration];
        for (i, &id) in gate_ids.iter().take(duration).enumerate() {
            text_gate[i] = id;
        }
        let mut text_neg = vec![-1i64; duration];
        for (i, &id) in neg_ids.iter().take(duration).enumerate() {
            text_neg[i] = id;
        }
        let positions: Vec<i64> = (0..duration as i64).collect();
        let mut cond = vec![0.0f32; duration * MEL_CH];
        cond[..ref_len * MEL_CH].copy_from_slice(&cond_mel);
        let zeros = vec![0.0f32; duration * MEL_CH];
        let mut x: Vec<f32> = (0..duration * MEL_CH).map(|_| normal.sample(rng)).collect();

        let run_step = |dit: &mut Session,
                        x: &[f32],
                        cd: &[f32],
                        txt: &[i64],
                        gate: &[i64],
                        t: f32,
                        moe: f32|
         -> Result<Vec<f32>> {
            let x_t = Tensor::from_array(([1usize, duration, MEL_CH], x.to_vec()))?;
            let c_t = Tensor::from_array(([1usize, duration, MEL_CH], cd.to_vec()))?;
            let txt_t = Tensor::from_array(([1usize, duration], txt.to_vec()))?;
            let gate_t = Tensor::from_array(([1usize, duration], gate.to_vec()))?;
            let time_t = Tensor::from_array(([1usize], vec![t]))?;
            let moe_t = Tensor::from_array(([1usize], vec![moe]))?;
            let pos_t = Tensor::from_array((vec![duration], positions.clone()))?;
            let out = dit.run(ort::inputs![
                "x" => x_t, "cond" => c_t, "text" => txt_t, "text_gate" => gate_t,
                "time" => time_t, "moe_flag" => moe_t, "positions" => pos_t
            ])?;
            let (_, data) = out["vel"].try_extract_tensor::<f32>()?;
            Ok(data.to_vec())
        };

        for i in 0..steps {
            let pred = run_step(dit, &x, &cond, &text, &text_gate, ts[i], 1.0)?;
            let vel = if args.cfg >= 1e-5 {
                let null = run_step(dit, &x, &zeros, &text_neg, &text_neg, ts[i], 0.0)?;
                pred.iter()
                    .zip(null.iter())
                    .map(|(p, n)| p + (p - n) * args.cfg)
                    .collect::<Vec<f32>>()
            } else {
                pred
            };
            let dt = ts[i + 1] - ts[i];
            for (xi, vi) in x.iter_mut().zip(vel.iter()) {
                *xi += dt * vi;
            }
        }

        // 条件段回填参考 mel（官方做法），取生成段
        x[..ref_len * MEL_CH].copy_from_slice(&cond_mel);
        let gen_frames = duration - ref_len;
        let mut gen_mel = vec![0.0f32; MEL_CH * gen_frames];
        for f in 0..gen_frames {
            for c in 0..MEL_CH {
                gen_mel[c * gen_frames + f] = x[(ref_len + f) * MEL_CH + c];
            }
        }
        Ok((gen_mel, gen_frames))
    };

    // mel → wav（Vocos + ISTFT）
    let mel_to_wav = |voc: &mut Session, gen_mel: Vec<f32>, gen_frames: usize| -> Result<Vec<f32>> {
        let mel_t = Tensor::from_array(([1usize, MEL_CH, gen_frames], gen_mel))?;
        let voc_out = voc.run(ort::inputs!["mel" => mel_t])?;
        let (_, spec) = voc_out["spec"].try_extract_tensor::<f32>()?;
        let bins = N_FFT / 2 + 1;
        let (re, im) = spec.split_at(bins * gen_frames);
        Ok(istft(re, im, gen_frames))
    };

    // mel 的能量突起（音节）计数：逐帧平均 log 能量 → 4 帧平滑 → 滞回阈值
    // （高=中位数+2.0，低=中位数）数上升沿。重复音节会多出一个突起、吞字
    // 少一个——与期望音节数差最小的候选最可能是干净的。
    let burst_count = |gen_mel: &[f32], gen_frames: usize| -> usize {
        let mut e: Vec<f32> = (0..gen_frames)
            .map(|f| {
                let s: f32 = (0..MEL_CH).map(|c| gen_mel[c * gen_frames + f]).sum();
                s / MEL_CH as f32
            })
            .collect();
        for i in 2..e.len() {
            e[i] = (e[i - 2] + e[i - 1] + e[i] * 2.0) / 4.0;
        }
        let mut sorted = e.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = sorted[sorted.len() / 2];
        let max = sorted[sorted.len() - 1];
        let range = (max - med).max(1e-3);
        let hi = med + 0.45 * range;
        let lo = med + 0.15 * range;
        let mut bursts = 0usize;
        let mut above = false;
        for v in &e {
            if !above && *v > hi {
                bursts += 1;
                above = true;
            } else if above && *v < lo {
                above = false;
            }
        }
        bursts
    };

    let t_ode = Instant::now();
    let mut chunk_wavs: Vec<Vec<f32>> = Vec::new();
    for (ci, chunk) in gen_chunks.iter().enumerate() {
        if args.best_of <= 1 {
            let (mel, gf) = synth_chunk(&mut dit, chunk, &mut rng)?;
            let wav = mel_to_wav(&mut voc, mel, gf)?;
            chunk_wavs.push(wav);
            continue;
        }
        // --best-of N：同块生成 N 个候选（独立 seed），双判据自动择优：
        // ①能量突起数 vs 期望音节数（主判据，直接对应重复/吞字失败模式）；
        // ②mel medoid 距离（同分时选最居中者）。零 ASR / 零人耳。
        let (syl, punct) = count_syllables(chunk);
        let expected = (syl + punct).max(1) as i64;
        let mut cands: Vec<(Vec<f32>, usize)> = Vec::new();
        for k in 0..args.best_of {
            let mut r = rand::rngs::StdRng::seed_from_u64(args.seed + (ci as u64) * 9973 + k as u64);
            cands.push(synth_chunk(&mut dit, chunk, &mut r)?);
        }
        let n = cands.len();
        let bursts: Vec<i64> = cands.iter().map(|(m, g)| burst_count(m, *g) as i64).collect();
        let mut dist = vec![0.0f64; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let (a, b) = (&cands[i].0, &cands[j].0);
                let m = a.len().min(b.len());
                let d: f64 = (0..m).step_by(7).map(|t| (a[t] - b[t]).abs() as f64).sum::<f64>()
                    / (m / 7 + 1) as f64;
                dist[i] += d;
                dist[j] += d;
            }
        }
        let best = (0..n)
            .min_by(|a, b| {
                // 主判据 medoid 距离（共识解）；平手时用突起数 vs 期望音节差
                dist[*a].partial_cmp(&dist[*b]).unwrap().then(
                    (bursts[*a] - expected).abs().cmp(&(bursts[*b] - expected).abs())
                )
            })
            .unwrap();
        eprintln!("  块 {}/{}: best-of {n} 选 #{best}（medoid 距 {:.4}，突起 {:?} vs 期望 {expected}）",
                  ci + 1, gen_chunks.len(), dist[best] / (n - 1) as f64, bursts);
        let (mel, gf) = cands.remove(best);
        chunk_wavs.push(mel_to_wav(&mut voc, mel, gf)?);
    }
    eprintln!("合成完成: {:.2}s", t_ode.elapsed().as_secs_f32());

    // ---- 交叉淡入淡出拼接（0.1s 余弦） ----
    let fade = (0.15 * SR as f32) as usize;
    let mut wav_out: Vec<f32> = Vec::new();
    for (i, w) in chunk_wavs.iter().enumerate() {
        if i == 0 {
            wav_out.extend_from_slice(w);
            continue;
        }
        let n = fade.min(wav_out.len()).min(w.len());
        // 尾部 fade-out 与新块头部 fade-in 叠加
        for j in 0..n {
            let a = (std::f32::consts::PI * (j as f32 / n as f32)).cos() * 0.5 + 0.5;
            let len = wav_out.len();
            wav_out[len - n + j] = wav_out[len - n + j] * a + w[j] * (1.0 - a);
        }
        wav_out.extend_from_slice(&w[n..]);
    }
    if chunk_wavs.len() == 1 {
        wav_out = chunk_wavs.into_iter().next().unwrap();
    }

    if out_scale != 1.0 {
        wav_out.iter_mut().for_each(|v| *v *= out_scale);
    }
    // 峰值保护：响度还原后仍超 ±0.95 时整体压低，避免 16bit 写出削波
    let peak_now = wav_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if peak_now > 0.95 {
        let g = 0.95 / peak_now;
        wav_out.iter_mut().for_each(|v| *v *= g);
    }
    // 输出 VAD 裁剪：去掉头尾多余静音/低能量杂音（阈值 3% 峰值，各留 80ms 余量）
    {
        let win = (SR as f32 * 0.02) as usize;
        if wav_out.len() > win * 2 {
            let peak = wav_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let thr = peak * 0.03;
            let n = wav_out.len() / win;
            let mut head = 0;
            while head < n
                && wav_out[head * win..(head + 1) * win].iter().all(|v| v.abs() < thr)
            {
                head += 1;
            }
            let mut tail = n;
            while tail > head
                && wav_out[(tail - 1) * win..tail.min(wav_out.len() / win) * win]
                    .iter()
                    .all(|v| v.abs() < thr)
            {
                tail -= 1;
            }
            let margin = (0.08 * SR as f32) as usize;
            let s = (head * win).saturating_sub(margin);
            let e = ((tail * win) + margin).min(wav_out.len());
            if e > s + win {
                wav_out = wav_out[s..e].to_vec();
            }
        }
        // 首尾微淡入淡出（5ms/15ms），消除裁剪边沿的爆音
        let fi = (0.005 * SR as f32) as usize;
        let fo = (0.015 * SR as f32) as usize;
        for i in 0..fi.min(wav_out.len()) {
            wav_out[i] *= i as f32 / fi as f32;
        }
        let n = wav_out.len();
        for i in 0..fo.min(n) {
            let j = n - 1 - i;
            wav_out[j] *= i as f32 / fo as f32;
        }
    }

    write_wav(&out_path, &wav_out)?;

    // ---- 统计输出 ----
    let audio_sec = wav_out.len() as f32 / SR as f32;
    let elapsed = t_start.elapsed().as_secs_f32();
    let out_rms = (wav_out.iter().map(|v| v * v).sum::<f32>() / wav_out.len().max(1) as f32).sqrt();
    let peak = wav_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!(
        "生成完成: {} | 生成音频 {:.2}s | 总耗时 {:.2}s | RTF {:.3} | RMS {:.5} | peak {:.4} | samples {}",
        out_path.display(),
        audio_sec,
        elapsed,
        elapsed / audio_sec.max(1e-9),
        out_rms,
        peak,
        wav_out.len()
    );

    Ok(())
}
