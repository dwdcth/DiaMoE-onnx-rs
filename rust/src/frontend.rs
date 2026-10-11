//! 普通话文本前端：汉字 → 无括号 IPA 串（再经 bracket_ipa 成模型输入）。
//!
//! 移植自官方 DiaMoE-TTS 的 dialect_frontend 管线（无 Python 依赖）：
//!   1. 拼音生成：jieba(posseg) 分词 + g2pw ONNX 多音字消歧 + pypinyin 词典
//!      数据（frontend_data/ 下的 TSV，含 FMM 词组匹配语义）
//!   2. 词级变调（tone_sandhi.py：一/不/轻声/合并，三三变调留给第 5 步）
//!   3. 儿化合并 + 词级读音纠正（polyphonic.tsv）
//!   4. word2dialect（#词# 标记 + 词级拼音替换，word_dialect.tsv）
//!   5. liandu_tone（(3,3)→(2,3) 与 一/不 规则，按标点分块、从后向前）
//!   6. 拼音→IPA（syllable_ipa.tsv）+ 声调符号（tone_mark.tsv）
//!
//! 与上游的差异（限制）：
//!   - 不做文本正则化（数字/英文请预先转成汉字；英文字母会被整体删除）；
//!   - 无 g2pM 兜底（g2pw 失败直接报错）；
//!   - 只内置普通话表；方言需导出对应表（管线相同）。

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use ort::value::Tensor;

const PUNC_ZH: &str = "、：，；。？！“”‘’':,;.?!…"; // zh_frontend.punc
const PUNC_TS: &str = "、：，；。？！“”‘’':,;.?!";   // tone_sandhi.punc
const NON_POLYPHONIC: &str = "一不和咋嗲剖差攢倒難奔勁拗肖瘙誒泊听噢";
const NON_MONOPHONIC: &str = "似攢";

fn load_lines(path: &Path) -> Result<Vec<String>> {
    Ok(std::fs::read_to_string(path)
        .with_context(|| format!("读取前端数据失败: {}", path.display()))?
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.trim_end_matches('\r').to_string())
        .collect())
}

fn load_set(path: &Path) -> Result<HashSet<String>> {
    Ok(load_lines(path)?.into_iter().collect())
}

fn tone_of(syl: &str) -> Option<u8> {
    syl.chars().last().and_then(|c| c.to_digit(10).map(|d| d as u8))
}

fn strip_last_digit(s: &str) -> String {
    if tone_of(s).is_some() {
        s.chars().take(s.chars().count() - 1).collect()
    } else {
        s.to_string()
    }
}

fn set_tone(finals: &mut [String], idx: usize, tone: u8) {
    if idx < finals.len() {
        finals[idx] = format!("{}{tone}", strip_last_digit(&finals[idx]));
    }
}

fn chars_of(s: &str) -> Vec<char> {
    s.chars().collect()
}

// ---------------- 拼音词典（pypinyin 数据 + FMM 词组匹配） ----------------

struct PinyinDict {
    chars: HashMap<char, Vec<String>>,
    phrases: HashMap<String, Vec<String>>,
    prefix_set: HashSet<String>,
}

impl PinyinDict {
    fn load(dir: &Path) -> Result<Self> {
        let mut chars = HashMap::new();
        for l in load_lines(&dir.join("chars.tsv"))? {
            let (c, rest) = l.split_once('\t').ok_or_else(|| anyhow!("chars.tsv 格式错: {l}"))?;
            let vs: Vec<String> = rest.split('|').map(|s| s.to_string()).collect();
            if let Some(ch) = c.chars().next() {
                if !vs.is_empty() {
                    chars.insert(ch, vs);
                }
            }
        }
        let mut phrases = HashMap::new();
        let mut prefix_set = HashSet::new();
        for l in load_lines(&dir.join("phrases.tsv"))? {
            let (w, rest) = l.split_once('\t').ok_or_else(|| anyhow!("phrases.tsv 格式错"))?;
            let pys: Vec<String> = rest.split(' ').map(|s| s.to_string()).collect();
            let wc: Vec<char> = w.chars().collect();
            for i in 1..=wc.len() {
                prefix_set.insert(wc[..i].iter().collect());
            }
            phrases.insert(w.to_string(), pys);
        }
        Ok(Self { chars, phrases, prefix_set })
    }

    /// 复刻 pypinyin.seg.mmseg.Seg.cut（正向最大匹配，no_non_phrases=False）。
    fn seg_cut(&self, text: &str) -> Vec<String> {
        let chars = chars_of(text);
        let mut out = Vec::new();
        let mut start = 0usize;
        while start < chars.len() {
            let rchars = &chars[start..];
            let mut matched = String::new();
            let mut last_valid: Option<(String, usize)> = None;
            let mut broke = false;
            for index in 0..rchars.len() {
                let word: String = rchars[..index + 1].iter().collect();
                if self.prefix_set.contains(&word) {
                    if self.phrases.contains_key(&word) {
                        last_valid = Some((word.clone(), index + 1));
                    }
                    matched = word;
                } else {
                    match &last_valid {
                        Some((w, idx)) => {
                            out.push(w.clone());
                            start += idx;
                        }
                        None => {
                            if !matched.is_empty() {
                                out.push(matched.clone());
                                start += matched.chars().count();
                            } else {
                                out.push(rchars[0].to_string());
                                start += 1;
                            }
                        }
                    }
                    broke = true;
                    break;
                }
            }
            if !broke {
                match last_valid {
                    Some((w, idx)) => {
                        out.push(w);
                        start += idx;
                    }
                    None => {
                        out.push(rchars.iter().collect());
                        break;
                    }
                }
            }
        }
        out
    }

    /// pypinyin(text, style=TONE3, neutral_tone_with_five=True) 等价。
    fn pinyin(&self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        for word in self.seg_cut(text) {
            if let Some(pys) = self.phrases.get(&word) {
                out.extend(pys.iter().cloned());
            } else {
                for c in word.chars() {
                    match self.chars.get(&c) {
                        Some(vs) => out.push(vs[0].clone()),
                        None => out.push(c.to_string()),
                    }
                }
            }
        }
        out
    }

    /// tone_sandhi 里 lazy_pinyin(word, FINALS_TONE3) 只用声调数字。
    fn tones(&self, word: &str) -> Vec<Option<u8>> {
        self.pinyin(word).iter().map(|s| tone_of(s)).collect()
    }
}

// ---------------- g2pw（多音字消歧，ONNX + HF tokenizers） ----------------

struct G2pw {
    session: ort::session::Session,
    out_name: String,
    tokenizer: tokenizers::Tokenizer,
    labels: Vec<String>,
    char2phonemes: HashMap<char, Vec<usize>>,
    chars_index: HashMap<char, usize>,
    polyphonic_new: HashSet<char>,
    monophonic: HashMap<char, String>,
}

impl G2pw {
    fn load(g2pw_dir: &Path) -> Result<Self> {
        let model = {
            let p = g2pw_dir.join("g2pw_int8.onnx");
            if p.exists() { p } else { g2pw_dir.join("g2pw.onnx") }
        };
        let session = ort::session::Session::builder()?
            .with_intra_threads(2)?
            .commit_from_file(&model)
            .with_context(|| format!("加载 g2pw 模型失败: {}", model.display()))?;
        let out_name = session.outputs[0].name.clone();
        let tokenizer = tokenizers::Tokenizer::from_file(
            g2pw_dir.join("bert-base-chinese").join("tokenizer.json"))
            .map_err(|e| anyhow!("加载 g2pw tokenizer 失败: {e}"))?;

        let mut pairs: Vec<(char, String)> = Vec::new();
        for l in load_lines(&g2pw_dir.join("POLYPHONIC_CHARS.txt"))? {
            let (c, p) = l.split_once('\t').ok_or_else(|| anyhow!("POLYPHONIC_CHARS 格式错"))?;
            if let Some(ch) = c.chars().next() {
                pairs.push((ch, p.to_string()));
            }
        }
        let mut labels: Vec<String> = pairs.iter().map(|(_, p)| p.clone()).collect();
        labels.sort();
        labels.dedup();
        let label_index: HashMap<&str, usize> =
            labels.iter().enumerate().map(|(i, l)| (l.as_str(), i)).collect();
        let mut char2phonemes: HashMap<char, Vec<usize>> = HashMap::new();
        for (c, p) in &pairs {
            if let Some(li) = label_index.get(p.as_str()) {
                char2phonemes.entry(*c).or_default().push(*li);
            }
        }
        let mut chars_list: Vec<char> = char2phonemes.keys().copied().collect();
        chars_list.sort();
        let chars_index: HashMap<char, usize> =
            chars_list.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let polyphonic_new: HashSet<char> =
            chars_list.iter().copied().filter(|c| !NON_POLYPHONIC.contains(*c)).collect();

        let b2p: HashMap<String, String> = serde_json::from_str(&std::fs::read_to_string(
            g2pw_dir.join("bopomofo_to_pinyin_wo_tune_dict.json"))?)?;
        let mut monophonic = HashMap::new();
        for l in load_lines(&g2pw_dir.join("MONOPHONIC_CHARS.txt"))? {
            let (c, b) = l.split_once('\t').ok_or_else(|| anyhow!("MONOPHONIC_CHARS 格式错"))?;
            let ch = match c.chars().next() { Some(x) => x, None => continue };
            if NON_MONOPHONIC.contains(ch) {
                continue;
            }
            let (body, tone) = b.split_at(b.len().saturating_sub(1));
            if let Some(base) = b2p.get(body) {
                monophonic.insert(ch, format!("{base}{tone}"));
            }
        }
        Ok(Self { session, out_name, tokenizer, labels, char2phonemes,
                  chars_index, polyphonic_new, monophonic })
    }

    /// wordize_and_map：英文/数字串成词，其余单字（返回 (词, 起始字符位, 结束字符位)）
    fn wordize(text_chars: &[char]) -> Vec<(String, usize, usize)> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < text_chars.len() {
            if text_chars[i].is_ascii_alphanumeric() {
                let start = i;
                while i < text_chars.len() && text_chars[i].is_ascii_alphanumeric() { i += 1; }
                out.push((text_chars[start..i].iter().collect(), start, i));
            } else {
                out.push((text_chars[i].to_string(), i, i + 1));
                i += 1;
            }
        }
        out
    }

    /// tokenize_and_map：逐词 WordPiece。返回 (tokens, text2token, token2text)。
    /// text2token 初始为 wordize 的词下标，被 token 覆盖的位置改写为 token 下标
    /// （与上游 Python 语义一致）。
    fn tokenize_and_map(&self, text_chars: &[char])
        -> Result<(Vec<String>, Vec<usize>, Vec<(usize, usize)>)> {
        let words = Self::wordize(text_chars);
        let mut text2token: Vec<usize> = vec![usize::MAX; text_chars.len()];
        for (wi, (word, ws, we)) in words.iter().enumerate() {
            for p in *ws..*we {
                text2token[p] = wi;
            }
        }
        let mut tokens = Vec::new();
        let mut token2text = Vec::new();
        for (word, ws, we) in &words {
            let enc = self.tokenizer.encode(word.as_str(), false)
                .map_err(|e| anyhow!("g2pw tokenize 失败: {e}"))?;
            let wt: Vec<String> = enc.get_tokens().to_vec();
            if wt.is_empty() || wt.len() == 1 && wt[0] == "[UNK]" {
                tokens.push("[UNK]".to_string());
                token2text.push((*ws, *we));
            } else {
                let mut cur = *ws;
                for t in wt {
                    let bare = t.strip_prefix("##").unwrap_or(&t).chars().count();
                    token2text.push((cur, cur + bare));
                    cur += bare;
                    tokens.push(t);
                }
            }
        }
        for (ti, (ts, te)) in token2text.iter().enumerate() {
            for p in *ts..*te {
                if p < text2token.len() {
                    text2token[p] = ti;
                }
            }
        }
        Ok((tokens, text2token, token2text))
    }

    /// G2PWOnnxConverter.__call__ 移植：整句 → 每字拼音（多音字走模型）。
    fn convert(&mut self, sent: &str, py: &PinyinDict, t2s: &HashMap<char, char>)
        -> Result<Vec<String>> {
        let sent_chars = chars_of(sent);
        let sent_s: Vec<char> = sent_chars.iter().map(|c| *t2s.get(c).unwrap_or(c)).collect();
        let sent_s_str: String = sent_s.iter().collect();
        let pypinyin_result = py.pinyin(&sent_s_str);

        let mut partial: Vec<Option<String>> = Vec::with_capacity(sent_chars.len());
        let mut query_ids: Vec<usize> = Vec::new();
        for (i, ch) in sent_s.iter().enumerate() {
            if self.polyphonic_new.contains(ch) {
                query_ids.push(i);
                partial.push(None);
            } else if let Some(p) = self.monophonic.get(ch) {
                partial.push(Some(p.clone()));
            } else {
                partial.push(Some(pypinyin_result.get(i).cloned()
                    .unwrap_or_else(|| ch.to_string())));
            }
        }
        if query_ids.is_empty() {
            return Ok(partial.into_iter().map(|o| o.unwrap()).collect());
        }

        // prepare_onnx_input（use_mask=true；模型输入文本用原句小写）
        let text_lower: Vec<char> = sent_chars.iter()
            .flat_map(|c| c.to_lowercase()).collect();
        let (tokens, text2token, _) = self.tokenize_and_map(&text_lower)?;
        let cls = self.tokenizer.token_to_id("[CLS]").ok_or_else(|| anyhow!("无 [CLS]"))? as i64;
        let sep = self.tokenizer.token_to_id("[SEP]").ok_or_else(|| anyhow!("无 [SEP]"))? as i64;
        let unk = self.tokenizer.token_to_id("[UNK]").ok_or_else(|| anyhow!("无 [UNK]"))? as i64;
        let mut ids = vec![cls];
        let mut masks = vec![1i64];
        for t in &tokens {
            ids.push(self.tokenizer.token_to_id(t).map(|v| v as i64).unwrap_or(unk));
            masks.push(1);
        }
        ids.push(sep);
        masks.push(1);
        let types = vec![0i64; ids.len()];

        let nl = self.labels.len();
        let mut phoneme_masks = Vec::with_capacity(query_ids.len() * nl);
        let mut char_ids = Vec::new();
        let mut position_ids = Vec::new();
        for &qid in &query_ids {
            let query_char = *sent_s.get(qid).ok_or_else(|| anyhow!("query 越界"))?;
            let valid: HashSet<usize> = self.char2phonemes.get(&query_char)
                .cloned().unwrap_or_default().into_iter().collect();
            for li in 0..nl {
                phoneme_masks.push(if valid.contains(&li) { 1.0f32 } else { 0.0 });
            }
            char_ids.push(*self.chars_index.get(&query_char)
                .ok_or_else(|| anyhow!("多音字 {query_char} 不在 chars 表"))? as i64);
            let tok_idx = text2token.get(qid).copied().unwrap_or(0);
            position_ids.push(tok_idx as i64 + 1);
        }

        let b = query_ids.len();
        // 每个多音字一行 batch，序列复制 b 份（上游 texts 每 query 一份同句）
        let rep = |v: &[i64]| -> Vec<i64> {
            (0..b).flat_map(|_| v.iter().copied()).collect()
        };
        // 输入名以模型为准（attention_mask / phoneme_mask 是单数，eps 是标量）
        let inputs = ort::inputs![
            "input_ids" => Tensor::from_array((vec![b, ids.len()], rep(&ids)))?,
            "token_type_ids" => Tensor::from_array((vec![b, types.len()], rep(&types)))?,
            "attention_mask" => Tensor::from_array((vec![b, masks.len()], rep(&masks)))?,
            "phoneme_mask" => Tensor::from_array((vec![b, nl], phoneme_masks))?,
            "char_ids" => Tensor::from_array((vec![b], char_ids.clone()))?,
            "position_ids" => Tensor::from_array((vec![b], position_ids.clone()))?,
            "eps" => Tensor::from_array((Vec::<usize>::new(), vec![1e-6f64]))?,
        ];
        let out = self.session.run(inputs)?;
        if std::env::var("DIAMOE_FE_DEBUG").is_ok() {
            eprintln!("[g2pw] qids={query_ids:?} char_ids={char_ids:?} pos={position_ids:?}");
            eprintln!("[g2pw] input_ids[0]={:?}", &rep(&ids)[..ids.len()]);
        }
        let (_, probs) = out[self.out_name.as_str()].try_extract_tensor::<f32>()?;
        for (bi, qid) in query_ids.iter().enumerate() {
            let row = &probs[bi * nl..(bi + 1) * nl];
            let best = row.iter().enumerate()
                .max_by(|a, c| a.1.partial_cmp(c.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            partial[*qid] = Some(self.labels[best].clone());
        }
        Ok(partial.into_iter().map(|o| o.unwrap()).collect())
    }
}

// ---------------- 前端主管线 ----------------

pub struct Frontend {
    py: PinyinDict,
    lexicon: HashMap<String, (String, String)>, // 音节 → (声母, 韵母含调)
    polyphonic_words: HashMap<String, String>,  // 词 → 读音（空格分隔）
    must_neural: HashSet<String>,
    must_not_neural: HashSet<String>,
    must_erhua: HashSet<String>,
    not_erhua: HashSet<String>,
    syllable_ipa: HashMap<String, String>, // pinyin → "ipa_initial,ipa_final"
    tone_marks: Vec<(String, String)>,
    word_dialect: Vec<(String, String)>,
    t2s: HashMap<char, char>,
    g2pw: G2pw,
    jieba: jieba_rs::Jieba,
}

impl Frontend {
    pub fn load(data_dir: &Path, g2pw_dir: &Path) -> Result<Self> {
        let py = PinyinDict::load(data_dir)?;
        let mut lexicon = HashMap::new();
        for l in load_lines(&data_dir.join("lexicon.tsv"))? {
            let (k, v) = l.split_once('\t').ok_or_else(|| anyhow!("lexicon.tsv 格式错"))?;
            let mut it = v.split(' ');
            let ini = it.next().unwrap_or("").to_string();
            let fin = it.next().unwrap_or("").to_string();
            lexicon.insert(k.to_string(), (ini, fin));
        }
        let polyphonic_words: HashMap<String, String> =
            load_lines(&data_dir.join("polyphonic.tsv"))?.iter()
                .filter_map(|l| l.split_once('\t').map(|(k, v)| (k.to_string(), v.to_string())))
                .collect();
        let mut syllable_ipa = HashMap::new();
        for l in load_lines(&data_dir.join("syllable_ipa.tsv"))? {
            let (k, v) = l.split_once('\t').ok_or_else(|| anyhow!("syllable_ipa.tsv 格式错"))?;
            syllable_ipa.insert(k.to_string(), v.to_string());
        }
        let tone_marks: Vec<(String, String)> =
            load_lines(&data_dir.join("tone_mark.tsv"))?.iter()
                .filter_map(|l| l.split_once('\t')
                    .map(|(a, b)| (a.to_string(), b.to_string()))).collect();
        let mut word_dialect = Vec::new();
        for l in load_lines(&data_dir.join("word_dialect.tsv"))? {
            if let Some((k, v)) = l.split_once('\t') {
                word_dialect.push((k.to_string(), v.to_string()));
            }
        }
        word_dialect.sort_by_key(|(w, _)| std::cmp::Reverse(w.replace('#', "").chars().count()));
        let mut t2s = HashMap::new();
        for l in load_lines(&data_dir.join("t2s.tsv"))? {
            if let Some((k, v)) = l.split_once('\t') {
                if let (Some(a), Some(b)) = (k.chars().next(), v.chars().next()) {
                    t2s.insert(a, b);
                }
            }
        }
        Ok(Self {
            py,
            lexicon,
            polyphonic_words,
            must_neural: load_set(&data_dir.join("must_neural.tsv"))?,
            must_not_neural: load_set(&data_dir.join("must_not_neural.tsv"))?,
            must_erhua: load_set(&data_dir.join("must_erhua.tsv"))?,
            not_erhua: load_set(&data_dir.join("not_erhua.tsv"))?,
            syllable_ipa,
            tone_marks,
            word_dialect,
            t2s,
            g2pw: G2pw::load(g2pw_dir)?,
            jieba: jieba_rs::Jieba::new(),
        })
    }

    // ---- word2dialect 的 tokenize 工具（#...# 单元感知） ----

    fn tokenize_text(text: &str) -> Vec<String> {
        // 等价 re.finditer(r"#([^#]+)#")：# 包裹段成单元，其余逐字
        let chars = chars_of(text);
        let mut out = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '#' {
                if let Some(end) = (i + 1..chars.len()).find(|&j| chars[j] == '#') {
                    let unit: String = chars[i..=end].iter().collect();
                    out.push(unit);
                    i = end + 1;
                    continue;
                }
            }
            out.push(chars[i].to_string());
            i += 1;
        }
        out
    }

    fn tokenize_pinyin(line: &str) -> Result<Vec<String>> {
        let mut result = Vec::new();
        let mut buffer: Vec<String> = Vec::new();
        let mut in_hash = false;
        for tok in line.split_whitespace() {
            let starts = tok.starts_with('#');
            let ends = tok.ends_with('#') && tok.len() > 1;
            if starts && !ends {
                in_hash = true;
                buffer = vec![tok.to_string()];
            } else if in_hash {
                buffer.push(tok.to_string());
                if tok.ends_with('#') {
                    result.push(buffer.join(" "));
                    in_hash = false;
                }
            } else {
                result.push(tok.to_string());
            }
        }
        if in_hash {
            bail!("pinyin 含未闭合的 #");
        }
        Ok(result)
    }

    // ---- tone_sandhi ----

    fn split_word(&self, word: &str) -> Vec<String> {
        let mut wl: Vec<String> = self.jieba.cut_for_search(word, true)
            .into_iter().map(|t| t.word.to_string()).collect();
        wl.sort_by_key(|s| s.chars().count());
        let first = wl.first().cloned().unwrap_or_default();
        if first.is_empty() {
            return vec![word.to_string()];
        }
        let wc = chars_of(word);
        let fc = chars_of(&first);
        if wc.len() >= fc.len() && wc[..fc.len()] == fc[..] {
            vec![first, wc[fc.len()..].iter().collect()]
        } else {
            let n = wc.len() - fc.len();
            vec![wc[..n].iter().collect(), first]
        }
    }

    fn neural_by_word(&self, word: &str) -> bool {
        if self.must_neural.contains(word) {
            return true;
        }
        let wc = chars_of(word);
        wc.len() >= 2 && {
            let tail: String = wc[wc.len() - 2..].iter().collect();
            self.must_neural.contains(&tail)
        }
    }

    fn neural_sandhi(&self, word: &str, pos: &str, finals: &mut [String]) {
        if self.must_not_neural.contains(word) {
            return;
        }
        let w = chars_of(word);
        for j in 1..w.len() {
            if w[j] == w[j - 1] && pos.starts_with(['n', 'v', 'a']) {
                set_tone(finals, j, 5);
            }
        }
        let last = *w.last().unwrap_or(&' ');
        let pos_in = |set: &[&str]| set.contains(&pos);
        if !w.is_empty() && "吧呢啊呐噻嘛吖嗨呐哦哒滴哩哟喽啰耶喔诶".contains(last) {
            set_tone(finals, w.len() - 1, 5);
        } else if !w.is_empty() && "的地得".contains(last) {
            set_tone(finals, w.len() - 1, 5);
        } else if w.len() == 1 && "了着过".contains(last)
            && pos_in(&["ul", "uz", "ug"]) {
            set_tone(finals, 0, 5);
        } else if w.len() > 1 && "们子".contains(last) && pos_in(&["r", "n"]) {
            set_tone(finals, w.len() - 1, 5);
        } else if w.len() > 1 && "上下".contains(last) && pos_in(&["s", "l", "f"]) {
            set_tone(finals, w.len() - 1, 5);
        } else if w.len() > 1 && "来去".contains(last)
            && "上下进出回过起开".contains(w[w.len() - 2]) {
            set_tone(finals, w.len() - 1, 5);
        } else {
            // 个 做量词（Python 的 str.isnumeric() 对中文数字（一二三百…）返回
            // True，这里用同一字符集近似）
            if let Some(gi) = w.iter().position(|c| *c == '个') {
                let cn_num = |c: char| c.is_ascii_digit()
                    || "零一二三四五六七八九十百千万亿两兆廿卅".contains(c);
                let prev_ok = gi >= 1 && (cn_num(w[gi - 1])
                    || "几有两半多各整每做是".contains(w[gi - 1]));
                if prev_ok || word == "个" {
                    set_tone(finals, gi, 5);
                } else if self.neural_by_word(word) {
                    set_tone(finals, w.len() - 1, 5);
                }
            } else if self.neural_by_word(word) {
                set_tone(finals, w.len() - 1, 5);
            }
        }
        // 子词级轻声
        let parts = self.split_word(word);
        let mut ofs = 0usize;
        for p in &parts {
            let n = p.chars().count();
            if !p.is_empty() && self.neural_by_word(p) {
                set_tone(finals, ofs + n - 1, 5);
            }
            ofs += n;
        }
    }

    fn bu_sandhi(word: &str, finals: &mut [String]) {
        let w = chars_of(word);
        if w.len() == 3 && w[1] == '不' {
            set_tone(finals, 1, 5);
        } else {
            for i in 0..w.len() {
                if w[i] == '不' && i + 1 < w.len() && tone_of(&finals[i + 1]) == Some(4) {
                    set_tone(finals, i, 2);
                }
            }
        }
    }

    fn yi_sandhi(word: &str, finals: &mut [String]) {
        let w = chars_of(word);
        let cn_num = |c: char| "零一二三四五六七八九十百千万亿两".contains(c);
        if w.contains(&'一') && w.iter().any(|c| *c != '一') && w.iter().all(|c| *c == '一' || cn_num(*c)) {
            return;
        }
        if w.len() == 3 && w[1] == '一' && w[0] == w[2] {
            set_tone(finals, 1, 5);
        } else if word.starts_with("第一") {
            set_tone(finals, 1, 1);
        } else {
            for i in 0..w.len() {
                if w[i] == '一' && i + 1 < w.len() {
                    match tone_of(&finals[i + 1]) {
                        Some(4) | Some(5) => set_tone(finals, i, 2),
                        _ => {
                            if !PUNC_TS.contains(w[i + 1]) {
                                set_tone(finals, i, 4);
                            }
                        }
                    }
                }
            }
        }
    }

    fn modified_tone(&self, word: &str, pos: &str, finals: &mut [String]) {
        Self::bu_sandhi(word, finals);
        Self::yi_sandhi(word, finals);
        self.neural_sandhi(word, pos, finals);
    }

    fn pre_merge_for_modify(&self, seg: &mut Vec<(String, String)>) {
        Self::merge_bu(seg);
        Self::merge_yi(seg);
        Self::merge_reduplication(seg);
        self.merge_three_tones(seg, false);
        self.merge_three_tones(seg, true);
        Self::merge_er(seg);
    }

    fn merge_bu(seg: &mut Vec<(String, String)>) {
        let mut new_seg: Vec<(String, String)> = Vec::new();
        let mut last_word = String::new();
        for (word, pos) in seg.drain(..) {
            let mut w = word;
            if last_word == "不" {
                w = format!("不{w}");
            }
            if w != "不" {
                new_seg.push((w.clone(), pos));
            }
            last_word = w;
        }
        if last_word == "不" {
            new_seg.push((last_word, "d".to_string()));
        }
        *seg = new_seg;
    }

    fn merge_yi(seg: &mut Vec<(String, String)>) {
        let mut new_seg: Vec<(String, String)> = Vec::new();
        for (i, (word, pos)) in seg.iter().enumerate() {
            if i >= 1 && word == "一" && i + 1 < seg.len()
                && seg[i - 1].0 == seg[i + 1].0 && seg[i - 1].1 == "v" {
                if i - 1 < new_seg.len() {
                    let base = new_seg[i - 1].0.clone();
                    new_seg[i - 1].0 = format!("{base}一{base}");
                } else {
                    new_seg.push((word.clone(), pos.clone()));
                    new_seg.push((seg[i + 1].0.clone(), pos.clone()));
                }
            } else if i >= 2 && seg[i - 1].0 == "一" && seg[i - 2].0 == *word {
                continue;
            } else {
                new_seg.push((word.clone(), pos.clone()));
            }
        }
        let mut seg2: Vec<(String, String)> = Vec::new();
        for (word, pos) in new_seg {
            if let Some(last) = seg2.last_mut() {
                if last.0 == "一" {
                    last.0 = format!("一{word}");
                    continue;
                }
            }
            seg2.push((word, pos));
        }
        *seg = seg2;
    }

    fn merge_reduplication(seg: &mut Vec<(String, String)>) {
        let mut new_seg: Vec<(String, String)> = Vec::new();
        for (word, pos) in seg.drain(..) {
            if let Some(last) = new_seg.last_mut() {
                if last.0 == word {
                    last.0.push_str(&word);
                    continue;
                }
            }
            new_seg.push((word, pos));
        }
        *seg = new_seg;
    }

    fn merge_three_tones(&self, seg: &mut Vec<(String, String)>, second: bool) {
        let tones: Vec<Vec<Option<u8>>> = seg.iter().map(|(w, _)| self.py.tones(w)).collect();
        let mut new_seg: Vec<(String, String)> = Vec::new();
        let mut merge_last = vec![false; seg.len()];
        for i in 0..seg.len() {
            let cond = if !second {
                i >= 1 && tones[i - 1].iter().all(|t| *t == Some(3))
                    && tones[i].iter().all(|t| *t == Some(3)) && !merge_last[i - 1]
            } else {
                i >= 1 && tones[i - 1].last().copied().flatten() == Some(3)
                    && tones[i].first().copied().flatten() == Some(3) && !merge_last[i - 1]
            };
            let redup = |w: &str| { let c = chars_of(w); c.len() == 2 && c[0] == c[1] };
            if cond && !redup(&seg[i - 1].0)
                && seg[i - 1].0.chars().count() + seg[i].0.chars().count() <= 3 {
                new_seg.last_mut().unwrap().0.push_str(&seg[i].0);
                merge_last[i] = true;
            } else {
                new_seg.push(seg[i].clone());
            }
        }
        *seg = new_seg;
    }

    fn merge_er(seg: &mut Vec<(String, String)>) {
        let mut new_seg: Vec<(String, String)> = Vec::new();
        for (word, pos) in seg.drain(..) {
            if word == "儿" && !new_seg.is_empty() {
                new_seg.last_mut().unwrap().0.push('儿');
            } else {
                new_seg.push((word, pos));
            }
        }
        *seg = new_seg;
    }

    // ---- 儿化（_merge_erhua） ----

    fn merge_erhua(&self, initials: &mut Vec<String>, finals: &mut Vec<String>,
                   word: &str, pos: &str) -> bool {
        let w = chars_of(word);
        // fix er1
        if w.len() == finals.len() && !w.is_empty() && *w.last().unwrap() == '儿'
            && finals.last().map(|f| f.as_str()) == Some("er1") {
            let n = finals.len();
            finals[n - 1] = "er2".to_string();
        }
        let pos_skip = pos == "a" || pos == "j" || pos == "nr";
        if !self.must_erhua.contains(word)
            && (self.not_erhua.contains(word) || pos_skip) {
            return false;
        }
        if finals.len() != w.len() {
            return false;
        }
        let mut new_initials: Vec<String> = Vec::new();
        let mut new_finals: Vec<String> = Vec::new();
        for i in 0..w.len() {
            let last2: String = if w.len() >= 2 {
                w[w.len() - 2..].iter().collect() } else { String::new() };
            let is_er_tail = i == w.len() - 1 && w[i] == '儿'
                && (finals[i] == "er2" || finals[i] == "er5")
                && !self.not_erhua.contains(&last2)
                && !new_finals.is_empty();
            if is_er_tail {
                let f = new_finals.last().unwrap().clone();
                let tone = f.chars().last().unwrap_or('5');
                *new_finals.last_mut().unwrap() =
                    format!("{}r{}", strip_last_digit(&f), tone);
            } else {
                new_initials.push(initials[i].clone());
                new_finals.push(finals[i].clone());
            }
        }
        let changed = new_finals.len() != finals.len()
            || new_finals.iter().zip(finals.iter()).any(|(a, b)| a != b);
        *initials = new_initials;
        *finals = new_finals;
        changed
    }

    // ---- 步骤 1-3：汉字 → (zhongwens, ppinyins) ----

    fn hanzi_to_ppinyin(&mut self, text: &str) -> Result<(Vec<String>, Vec<String>)> {
        // 删除英文字母（与上游 re.sub("[a-zA-Z]+", "", seg) 一致；无正则化）
        let seg: String = text.chars().filter(|c| !c.is_ascii_alphabetic()).collect();
        if seg.is_empty() {
            bail!("输入没有有效汉字");
        }
        let mut seg_cut: Vec<(String, String)> = self.jieba
            .tag(&seg, true).into_iter()
            .map(|t| (t.word.to_string(), t.tag.to_string())).collect();
        let mut seg_cut = seg_cut;
        self.pre_merge_for_modify(&mut seg_cut);

        // 整句 g2pw 拼音。注意：上游 TextNormalizer 会把 CJK 标点转成 ASCII
        // （，→, 。→. 等）后才喂 g2pw，模型对两种标点的 token id 不同、预测
        // 可能不同（实测「地」di5/di4 会翻）——这里对齐该行为，只作用于
        // 模型输入文本，不改变下游输出（下游保持 CJK 标点，与官方产物一致）。
        const PUNCT_C2A: &[(&str, &str)] = &[("，", ","), ("。", "."), ("！", "!"), ("？", "?"),
            ("：", ":"), ("；", ";"), ("“", "\""), ("”", "\""), ("‘", "'"), ("’", "'"),
            ("（", "("), ("）", ")"), ("【", "["), ("】", "]"), ("《", "<"), ("》", ">")];
        let mut seg_model = seg.clone();
        for (c, a) in PUNCT_C2A {
            seg_model = seg_model.replace(c, a);
        }
        let pinyins_all = self.g2pw.convert(&seg_model, &self.py, &self.t2s)
            .context("g2pw 多音字消歧失败")?;
        // 模型输入用的 ASCII 标点透传单元转回 CJK（对应上游
        // pinyin2ipa 的 replace_english_punctuation，最终产物保持 CJK 标点）
        let pinyins_all: Vec<String> = pinyins_all.into_iter().map(|p| {
            match p.as_str() {
                "," => "，".to_string(),
                "." => "。".to_string(),
                "!" => "！".to_string(),
                "?" => "？".to_string(),
                ":" => "：".to_string(),
                ";" => "；".to_string(),
                _ => p,
            }
        }).collect();
        // 符号相连拆分
        let mut pinyins: Vec<String> = Vec::new();
        for p in &pinyins_all {
            let n = p.chars().count();
            if n <= 1 {
                pinyins.push(p.clone());
            } else if p.chars().all(|c| PUNC_ZH.contains(c)) {
                for c in p.chars() {
                    pinyins.push(c.to_string());
                }
            } else {
                pinyins.push(p.clone());
            }
        }

        let mut ppinyins: Vec<String> = Vec::new();
        let mut zhongwens: Vec<String> = Vec::new();
        let mut pre_len = 0usize;
        for (word, pos) in &seg_cut {
            let now_len = pre_len + word.chars().count();
            if pos == "eng" {
                pre_len = now_len;
                continue;
            }
            // corrector
            let corrected: Option<Vec<String>> = self.polyphonic_words.get(word)
                .map(|v| v.split(' ').map(|s| s.to_string()).collect());
            let word_pinyins: Vec<String> = match corrected {
                Some(v) => v,
                None => pinyins.get(pre_len..now_len).map(|s| s.to_vec())
                    .unwrap_or_else(|| vec![word.clone()]),
            };
            let wc = chars_of(word);
            let mut sub_pinyins: Vec<String> = Vec::new();
            let mut sub_initials: Vec<String> = Vec::new();
            let mut sub_finals: Vec<String> = Vec::new();
            for (pinyin, ch) in word_pinyins.iter().zip(wc.iter()) {
                let pinyin = pinyin.replace("u:", "v");
                let pinyin = if pinyin.is_empty() { ch.to_string() } else { pinyin };
                sub_pinyins.push(pinyin.clone());
                if let Some((ini, fin)) = self.lexicon.get(&pinyin) {
                    sub_initials.push(ini.clone());
                    sub_finals.push(fin.clone());
                } else {
                    sub_initials.push(pinyin.clone());
                    sub_finals.push(pinyin);
                }
            }
            pre_len = now_len;

            self.modified_tone(word, pos, &mut sub_finals);
            let erhua = self.merge_erhua(&mut sub_initials, &mut sub_finals, word, pos);

            // 声调写回拼音
            let mut sub_zhongwens: Vec<String> = wc.iter().map(|c| c.to_string()).collect();
            for i in 0..sub_finals.len().min(sub_pinyins.len()) {
                if sub_finals[i] != "None" && !PUNC_ZH.contains(
                    sub_finals[i].chars().next().unwrap_or(' ')) {
                    sub_pinyins[i] = format!("{}{}",
                        strip_last_digit(&sub_pinyins[i]),
                        tone_of(&sub_finals[i]).unwrap_or(5));
                }
            }
            if erhua {
                let n = sub_pinyins.len();
                let last_py = format!("#{}{}#", sub_initials.last().map(|s| s.as_str()).unwrap_or(""),
                                      sub_finals.last().map(|s| s.as_str()).unwrap_or(""));
                let zh_tail: String = sub_zhongwens[n.saturating_sub(2)..].concat();
                sub_pinyins.truncate(n.saturating_sub(2));
                sub_pinyins.push(last_py);
                sub_zhongwens.truncate(n.saturating_sub(2));
                sub_zhongwens.push(format!("#{zh_tail}#"));
            }
            ppinyins.extend(sub_pinyins);
            zhongwens.extend(sub_zhongwens);
        }
        Ok((zhongwens, ppinyins))
    }

    // ---- 步骤 4：word2dialect ----

    fn word2dialect(&self, zhongwens: &[String], ppinyins: &[String])
        -> Result<(Vec<String>, Vec<String>)> {
        let hash_only: Vec<&(String, String)> = self.word_dialect.iter()
            .filter(|(w, _)| w.contains('#')).collect();
        let mut text_units = Self::tokenize_text(&zhongwens.concat());
        let mut pinyin_units = Self::tokenize_pinyin(&ppinyins.join(" "))?;
        if text_units.len() != pinyin_units.len() {
            bail!("词表标记阶段 text/pinyin 不对齐: {} vs {}", text_units.len(), pinyin_units.len());
        }
        // insert_hash_marks_aligned：最长优先加 #...#
        let mut i = 0;
        while i < text_units.len() {
            let mut hit = None;
            for (w, _) in &hash_only {
                let plain = w.replace('#', "");
                let l = plain.chars().count();
                if i + l > text_units.len() { continue; }
                let segment = text_units[i..i + l].concat();
                if segment == plain {
                    hit = Some((l, plain, w.clone()));
                    break;
                }
            }
            if let Some((l, plain, _)) = hit {
                let joined_py = pinyin_units[i..i + l].join(" ");
                text_units.splice(i..i + l,
                    std::iter::once(format!("#{plain}#"))
                        .chain(std::iter::repeat(String::new()).take(l - 1)));
                pinyin_units.splice(i..i + l,
                    std::iter::once(format!("#{joined_py}#"))
                        .chain(std::iter::repeat(String::new()).take(l - 1)));
                i += l - 1;
            }
            i += 1;
        }
        let text_units: Vec<String> = text_units.into_iter().filter(|s| !s.is_empty()).collect();
        let mut pinyin_units: Vec<String> = pinyin_units.into_iter().filter(|s| !s.is_empty()).collect();

        // apply_replacements（词级替换，最长优先）
        let mut i = 0;
        while i < text_units.len() {
            for (word, rep) in &self.word_dialect {
                let wu = Self::tokenize_text(word);
                let m = wu.len();
                if i + m <= text_units.len() && text_units[i..i + m] == wu[..] {
                    let keep = rep.clone();
                    pinyin_units.splice(i..i + m,
                        std::iter::once(keep)
                            .chain(std::iter::repeat(String::new()).take(m - 1)));
                    i += m - 1;
                    break;
                }
            }
            i += 1;
        }
        Ok((text_units, pinyin_units.into_iter().filter(|s| !s.is_empty()).collect()))
    }

    // ---- 步骤 5：liandu_tone ----

    fn parse_tone(unit: &str) -> (String, Option<u8>) {
        let s = unit.trim_start_matches('#').trim_end_matches('#');
        let base: String = s.chars().take_while(|c| !c.is_ascii_digit() && *c != '#').collect();
        let tone = tone_of(s);
        (base, tone)
    }

    fn rewrite_unit(unit: &str, new_tone: u8) -> String {
        let (base, _) = Self::parse_tone(unit);
        if unit.starts_with('#') {
            format!("#{base}{new_tone}#")
        } else {
            format!("{base}{new_tone}")
        }
    }

    fn liandu_tone(&self, text_units: &[String], pinyin_units: &[String]) -> Vec<String> {
        const SEPS: &str = "，。！？,.!?;；";
        let sep_unit = |u: &String| u.chars().any(|c| SEPS.contains(c));
        let mut result: Vec<String> = Vec::new();
        let mut block_py: Vec<String> = Vec::new();
        let mut block_tx: Vec<String> = Vec::new();
        for (i, u) in pinyin_units.iter().enumerate() {
            if sep_unit(&text_units[i]) {
                result.extend(self.tone_change_block(&block_tx, &block_py));
                result.push(u.clone());
                block_py.clear();
                block_tx.clear();
            } else {
                block_py.push(u.clone());
                block_tx.push(text_units[i].clone());
            }
        }
        result.extend(self.tone_change_block(&block_tx, &block_py));
        result
    }

    fn tone_change_block(&self, block_tx: &[String], block: &[String]) -> Vec<String> {
        let mut r: Vec<String> = block.to_vec();
        // (3,3)→(2,3)：从后向前，匹配后回退
        let mut i = r.len() as i64 - 1;
        while i >= 0 {
            let start = i - 1;
            if start >= 0 {
                let t0 = Self::parse_tone(&r[start as usize]).1;
                let t1 = Self::parse_tone(&r[i as usize]).1;
                if t0 == Some(3) && t1 == Some(3) {
                    r[start as usize] = Self::rewrite_unit(&r[start as usize], 2);
                    i = start - 1;
                    continue;
                }
            }
            i -= 1;
        }
        // 一/不 字规则：从后向前，按 text_units 的字判断（上游 CHAR_TONE_RULES
        // 键是汉字不是拼音）
        let rules: &[(&str, &[(Option<u8>, u8)], u8)] = &[
            ("一", &[(Some(1), 4), (Some(2), 4), (Some(3), 4), (Some(4), 2)], 1),
            ("不", &[(Some(4), 2)], 4),
        ];
        let n = r.len();
        for idx in (0..n.saturating_sub(1)).rev() {
            let unit_char = block_tx.get(idx)
                .map(|u| u.trim_matches('#').to_string())
                .unwrap_or_default();
            for (ch, conds, default) in rules {
                if unit_char == *ch {
                    let next_tone = Self::parse_tone(&r[idx + 1]).1;
                    let mut new_tone = *default;
                    for (nt, v) in conds.iter() {
                        if *nt == next_tone {
                            new_tone = *v;
                            break;
                        }
                    }
                    r[idx] = Self::rewrite_unit(&r[idx], new_tone);
                    break;
                }
            }
        }
        r
    }

    // ---- 步骤 6：拼音→IPA + 声调符号 ----

    fn pinyin_to_ipa(&self, pinyin_units: &[String]) -> String {
        let ipa_units: Vec<String> = pinyin_units.iter().map(|u| {
            let base = u.trim_matches('#');
            self.syllable_ipa.get(base).cloned().unwrap_or_else(|| base.to_string())
        }).collect();
        let joined = ipa_units.join(" ");
        let s = joined.replace(' ', " | ").replace(',', " ");
        let mut s = s;
        for (contour, mark) in &self.tone_marks {
            s = s.replace(contour.as_str(), mark.as_str());
        }
        s
    }

    /// 完整管线：汉字 → 无括号 IPA（空格分隔，| 分隔音节）。
    pub fn hanzi_to_ipa(&mut self, text: &str) -> Result<String> {
        let (zhongwens, ppinyins) = self.hanzi_to_ppinyin(text)?;
        if std::env::var("DIAMOE_FE_DEBUG").is_ok() {
            eprintln!("[fe] zh=[{}] py=[{}]", zhongwens.concat(), ppinyins.join(" "));
        }
        let (text_units, pinyin_units) = self.word2dialect(&zhongwens, &ppinyins)?;
        let toned = self.liandu_tone(&text_units, &pinyin_units);
        Ok(self.pinyin_to_ipa(&toned))
    }
}
