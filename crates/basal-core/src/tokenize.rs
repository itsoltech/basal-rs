//! Hugging Face tokenizer of the checkpoint and the letter-id check of upstream `prompt.letter_ids`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, ensure, Context, Result};
use std::str::FromStr;

use crate::prompt::{ChatStyle, PromptFormat};
use crate::weights::GgufSemanticError;

#[derive(Debug, thiserror::Error)]
pub(crate) enum TokenizerInputError {
    #[error(transparent)]
    Semantic(#[from] anyhow::Error),
    #[error("tokenizer backend construction failed")]
    Backend {
        path: PathBuf,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl TokenizerInputError {
    fn into_legacy(self) -> anyhow::Error {
        match self {
            Self::Semantic(source) => source,
            Self::Backend { path, source } => anyhow!("{}: {source}", path.display()),
        }
    }
}

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
        Self::from_json_str(&raw, path, style).map_err(TokenizerInputError::into_legacy)
    }

    pub(crate) fn from_json_str(raw: &str, path: &Path, style: ChatStyle) -> Result<Self, TokenizerInputError> {
        let json = serde_json::from_str(raw).map_err(anyhow::Error::from)?;
        Self::from_json_value(json, path, style)
    }

    pub(crate) fn from_json_value(
        mut json: serde_json::Value,
        path: &Path,
        style: ChatStyle,
    ) -> Result<Self, TokenizerInputError> {
        Self::normalize_json(&mut json, style)?;
        let backend = |source| TokenizerInputError::Backend { path: path.to_path_buf(), source };
        let mut inner = tokenizers::Tokenizer::from_str(&json.to_string()).map_err(backend)?;
        // Upstream encodes without truncation, including when the serialized file enables it.
        inner.with_truncation(None).map_err(backend)?;
        inner.with_padding(None);
        Ok(Self { inner, letters: Mutex::new(HashMap::new()) })
    }

    fn normalize_json(json: &mut serde_json::Value, style: ChatStyle) -> Result<()> {
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
        Ok(())
    }

    /// Audit the original ID storage before normalization or dependency deserialization can replace it.
    pub(crate) fn audit_raw_ids(
        original: &serde_json::Value,
        vocab: usize,
    ) -> Result<Vec<(String, u32)>, GgufSemanticError> {
        use serde_json::Value;
        let invalid = |location| GgufSemanticError::TokenId { location, id: None, vocab };
        let model = original.get("model").ok_or_else(|| invalid("raw model"))?;
        match model.get("type").and_then(Value::as_str) {
            Some("BPE" | "WordPiece" | "WordLevel") => {
                let ids = model.get("vocab").and_then(Value::as_object).ok_or_else(|| invalid("raw model vocab"))?;
                for id in ids.values() {
                    GgufSemanticError::parse_id(id, "raw model vocab", vocab)?;
                }
            }
            Some("Unigram") => {
                let entries =
                    model.get("vocab").and_then(Value::as_array).ok_or_else(|| invalid("raw Unigram vocab"))?;
                for id in 0..entries.len() {
                    let id = u64::try_from(id).map_err(|_| invalid("raw Unigram index"))?;
                    GgufSemanticError::check_id(id, "raw Unigram index", vocab)?;
                }
                if let Some(id) = model.get("unk_id").filter(|v| !v.is_null()) {
                    let id = GgufSemanticError::parse_id(id, "raw Unigram unk_id", vocab)?;
                    if usize::try_from(id).ok().is_none_or(|id| id >= entries.len()) {
                        return Err(invalid("raw Unigram unk_id outside vocab"));
                    }
                }
            }
            _ => return Err(invalid("raw model ID representation")),
        }
        let mut added = Vec::new();
        let mut declared_ids = HashMap::new();
        let mut declared_contents = HashMap::new();
        if let Some(tokens) = original.get("added_tokens") {
            for token in tokens.as_array().ok_or_else(|| invalid("raw added_tokens"))? {
                let id = token.get("id").ok_or_else(|| invalid("raw added token ID"))?;
                let id = GgufSemanticError::parse_id(id, "raw added token ID", vocab)?;
                let content =
                    token.get("content").and_then(Value::as_str).ok_or_else(|| invalid("raw added content"))?;
                if declared_ids.insert(id, content).is_some_and(|previous| previous != content)
                    || declared_contents.insert(content, id).is_some_and(|previous| previous != id)
                {
                    return Err(invalid("conflicting raw added IDs"));
                }
                added.push((content.to_owned(), id));
            }
        }
        Ok(added)
    }

    pub(crate) fn audit_effective_ids(
        &self,
        added: &[(String, u32)],
        vocab: usize,
        bos: &str,
    ) -> Result<u32, GgufSemanticError> {
        use tokenizers::Model;
        // Inspect the base map separately: an added-token overlay can hide a base ID.
        for id in self.inner.get_model().get_vocab().values() {
            GgufSemanticError::check_id(u64::from(*id), "effective model vocab", vocab)?;
        }
        for id in self.inner.get_added_vocabulary().get_vocab().values() {
            GgufSemanticError::check_id(u64::from(*id), "effective added vocab", vocab)?;
        }
        for id in self.inner.get_added_vocabulary().get_added_tokens_decoder().keys() {
            GgufSemanticError::check_id(u64::from(*id), "effective added decoder", vocab)?;
        }
        for (content, id) in added {
            if self.token_id(content) != Some(*id) {
                return Err(GgufSemanticError::TokenId {
                    location: "added ID reconstruction",
                    id: Some(u64::from(*id)),
                    vocab,
                });
            }
        }
        let id =
            self.token_id(bos).ok_or(GgufSemanticError::TokenId { location: "BOS membership", id: None, vocab })?;
        GgufSemanticError::check_id(u64::from(id), "BOS", vocab)
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

    /// Decode a complete-character token slice, retaining special tokens like upstream `tok.decode`.
    pub(crate) fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids, false).map_err(|e| anyhow!("tokenizer decode: {e}"))
    }

    /// Decode generated notes with vLLM's default special-token exclusion; truncation keeps using `decode`.
    pub(crate) fn decode_thinking(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids, true).map_err(|e| anyhow!("tokenizer thinking decode: {e}"))
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub(crate) fn has_dense_vocabulary(&self) -> bool {
        let vocab = self.inner.get_vocab(true);
        let mut ids: Vec<u32> = vocab.values().copied().collect();
        ids.sort_unstable();
        ids.iter().enumerate().all(|(i, id)| usize::try_from(*id) == Ok(i))
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
        let res = self.context_code_ids(fmt, prompt, base, k)?;
        self.letters.lock().map_err(|_| anyhow!("tokenizer code cache poisoned"))?.insert((fmt, k), res.clone());
        Ok(res)
    }

    /// Validate codes in this exact answer context, bypassing the ordinary format/count cache (thinking P2).
    pub fn context_code_ids(
        &self,
        fmt: PromptFormat,
        prompt: &str,
        base: &[u32],
        k: usize,
    ) -> Result<Option<Vec<u32>>> {
        ensure!((2..=fmt.max_options()).contains(&k), "unsupported option count {k}");
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
        Ok((ok && uniq.len() == k).then_some(ids))
    }

    /// Conservative decoded-byte bound for the validated Llama Metaspace/ByteFallback decoder: replacing an
    /// invalid byte can emit three UTF-8 bytes. Include a space per piece, even when the decoder removes it.
    pub(crate) fn llama_decoded_token_bytes(&self) -> Result<usize> {
        let decoder = serde_json::json!({"type":"Sequence","decoders":[
            {"type":"Replace","pattern":{"String":"▁"},"content":" "},
            {"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]});
        ensure!(
            serde_json::to_value(self.inner.get_decoder())? == decoder,
            "thinking decoded-byte bound requires the published Llama decoder"
        );
        Ok(self.inner.get_vocab(true).keys().map(|s| s.len().saturating_mul(3).saturating_add(1)).max().unwrap_or(0))
    }
}
