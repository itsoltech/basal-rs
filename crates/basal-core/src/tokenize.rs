//! Hugging Face tokenizer of the checkpoint and the letter-id check of upstream `prompt.letter_ids`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, ensure, Context, Result};
use std::str::FromStr;

use crate::prompt::{ChatStyle, PromptFormat};

type CodeCache = HashMap<(PromptFormat, usize), Option<Vec<u32>>>;

pub struct BasalTokenizer {
    inner: tokenizers::Tokenizer,
    letters: Mutex<CodeCache>,
}

/// Character offsets (start, end) of every token.
pub type Offsets = Vec<(usize, usize)>;

impl BasalTokenizer {
    /// Loads `tokenizer.json` as upstream sees it through `AutoTokenizer`. transformers 5 `LlamaTokenizer` rebuilds the
    /// pre-tokenizer as `Metaspace(prepend_scheme="first")` (legacy=False, add_prefix_space=True) while the file
    /// stores `"always"`; vocabulary, merges, normalizer and the rest are taken from the file unchanged (checked
    /// against transformers 5.17.0). Any other pre-tokenizer is rejected rather than guessed.
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_for(path, ChatStyle::Llama)
    }

    pub fn load_for(path: &Path, style: ChatStyle) -> Result<Self> {
        let raw = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut json: serde_json::Value = serde_json::from_str(&raw)?;
        if matches!(style, ChatStyle::Gemma4 | ChatStyle::Gemma4Unified) {
            ensure!(
                json.get("pre_tokenizer")
                    == Some(
                        &serde_json::json!({"type":"Split", "pattern":{"String":" "}, "behavior":"MergedWithPrevious", "invert":false})
                    ),
                "unsupported Gemma pre-tokenizer"
            );
            ensure!(
                json.get("normalizer")
                    == Some(&serde_json::json!({"type":"Replace", "pattern":{"String":" "}, "content":"▁"})),
                "unsupported Gemma normalizer"
            );
        } else if style == ChatStyle::Qwen35 {
            // Qwen's NFC + byte-level BPE is used as shipped, with no Llama Metaspace rewrite.
            ensure!(
                json.get("normalizer") == Some(&serde_json::json!({"type":"NFC"})),
                "unsupported Qwen3.5 normalizer"
            );
            ensure!(
                json.pointer("/pre_tokenizer/type").and_then(|v| v.as_str()) == Some("Sequence")
                    && json.pointer("/pre_tokenizer/pretokenizers/1/type").and_then(|v| v.as_str())
                        == Some("ByteLevel"),
                "unsupported Qwen3.5 pre-tokenizer"
            );
        } else {
            // basal-1.0 ships "always" and transformers 5 runs it as "first"; basal-1.5 ships "first".
            let pre = json.get_mut("pre_tokenizer").context("tokenizer.json: pre_tokenizer missing")?;
            let scheme = pre.get("prepend_scheme").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let mut want = serde_json::json!({"type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": "first", "split": false});
            want["prepend_scheme"] = scheme.clone().into();
            ensure!(
                *pre == want && (scheme == "always" || scheme == "first"),
                "tokenizer.json: unsupported pre_tokenizer {pre}"
            );
            pre["prepend_scheme"] = "first".into();
            ensure!(json.get("normalizer").is_none_or(|n| n.is_null()), "tokenizer.json: unexpected normalizer");
        }
        let mut inner =
            tokenizers::Tokenizer::from_str(&json.to_string()).map_err(|e| anyhow!("{}: {e}", path.display()))?;
        // basal-1.0's tokenizer.json carries truncation at 1536 tokens; upstream encodes without truncation
        // (transformers disables it for calls without `truncation=`), so prompts are never cut here either.
        inner.with_truncation(None).map_err(|e| anyhow!("{}: {e}", path.display()))?;
        inner.with_padding(None);
        Ok(Self { inner, letters: Mutex::new(HashMap::new()) })
    }

    /// `tok(text, add_special_tokens=False).input_ids`
    /// Token ids and character offsets (Python string indices) of `text`, without special tokens.
    pub fn encode_char_offsets(&self, text: &str) -> Result<(Vec<u32>, Offsets)> {
        let enc = self.inner.encode_char_offsets(text, false).map_err(|e| anyhow!("tokenizer: {e}"))?;
        Ok((enc.get_ids().to_vec(), enc.get_offsets().to_vec()))
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.inner.encode(text, false).map_err(|e| anyhow!("tokenizer: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Token id of each of the first `k` letters at the answer position; `None` when a letter is not a single new
    /// token there or two letters share an id. Cached per `k` like upstream `Server.jobs_for`: the answer position
    /// always follows the same prefill, so the first prompt with `k` options decides for all.
    pub fn letter_ids(&self, prompt: &str, base: &[u32], k: usize) -> Result<Option<Vec<u32>>> {
        self.code_ids(PromptFormat::One, prompt, base, k)
    }

    /// Model-format option code ids, checked in the full answer context and cached separately by format.
    pub fn code_ids(&self, fmt: PromptFormat, prompt: &str, base: &[u32], k: usize) -> Result<Option<Vec<u32>>> {
        ensure!((2..=fmt.max_options()).contains(&k), "unsupported option count {k}");
        if let Some(hit) = self.letters.lock().map_err(|_| anyhow!("tokenizer code cache poisoned"))?.get(&(fmt, k)) {
            return Ok(hit.clone());
        }
        let mut ids = Vec::with_capacity(k);
        let mut ok = true;
        for l in fmt.codes().iter().take(k) {
            let full = self.encode(&format!("{prompt}{l}"))?;
            if full.len() != base.len() + 1 || full[..base.len()] != *base {
                ok = false;
                break;
            }
            ids.push(*full.last().unwrap());
        }
        let mut uniq = ids.clone();
        uniq.sort_unstable();
        uniq.dedup();
        let res = (ok && uniq.len() == k).then_some(ids);
        self.letters.lock().map_err(|_| anyhow!("tokenizer code cache poisoned"))?.insert((fmt, k), res.clone());
        Ok(res)
    }
}
