//! Opt-in admission of the tokenizer BPE builder's merge scratch buffer.
//!
//! Only that temporary buffer is admitted. Tokenizer JSON values, serde helper
//! storage, model maps and strings, added vocabulary, native regex state and
//! caches remain unadmitted, as on the original route.

use std::path::Path;

use tokenizers::utils::merge_storage::{MergeBuildError, MergeStorage};

use super::immutable_semantics::GgufImmutableSemanticError;
use super::raw_source::RawCharge;
use super::semantics::{build_captured_tokenizer_with, tokenizer_input_error};
use super::{GgufCompanionSnapshot, GgufRawSourceBudget, GgufRawSourceError, GgufSemanticError};
use crate::prompt::ChatStyle;
use crate::tokenize::{BasalTokenizer, TokenizerInputError};

/// Typed refusal of the BPE merge scratch buffer. No input text is retained.
#[derive(Debug, thiserror::Error)]
pub enum GgufMergeScratchError {
    #[error("BPE merge scratch admission failed")]
    Admission {
        #[source]
        source: GgufRawSourceError,
    },
    #[error("BPE merge scratch buffer has {actual} bytes, but {requested} were requested")]
    ProviderLength { requested: usize, actual: usize },
}

/// Failure of `GgufImmutableCompanionSemantics::read_with_bpe_merge_scratch_budget`.
#[derive(Debug, thiserror::Error)]
pub enum GgufMergeScratchReadError {
    /// Any failure of the existing immutable read, unchanged.
    #[error("GGUF immutable companion read failed")]
    Existing {
        #[from]
        source: GgufImmutableSemanticError,
    },
    #[error("GGUF tokenizer BPE merge scratch was refused")]
    Scratch {
        #[source]
        source: GgufMergeScratchError,
    },
}

/// Per-build scratch storage on the declaration budget's raw ledger.
struct MergeScratch<'a> {
    raw: &'a GgufRawSourceBudget,
}

/// Exact zeroed scratch bytes. Fields drop in order: the backing deallocates
/// before its charge refunds, on success, error and unwind alike.
struct MergeScratchBuffer {
    data: Vec<u8>,
    _charge: RawCharge,
}

impl AsMut<[u8]> for MergeScratchBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

impl MergeStorage for MergeScratch<'_> {
    type Buffer = MergeScratchBuffer;
    type Error = GgufRawSourceError;

    /// Checks the layout and acquires its charge before allocating. An empty
    /// buffer neither allocates nor charges. Refusals are fixed variants.
    fn try_zeroed(&self, length: usize) -> Result<MergeScratchBuffer, GgufRawSourceError> {
        // Declared first, so the charge outlives `data` if anything unwinds.
        let mut charge = self.raw.charge();
        let mut data = charge.allocate(length)?;
        // Initializes the exact capacity; it never reallocates.
        data.resize(length, 0);
        Ok(MergeScratchBuffer { data, _charge: charge })
    }
}

/// Private union of the existing semantic channel and the scratch refusal.
enum ScratchTokenizerError {
    Semantic(GgufSemanticError),
    Scratch(GgufMergeScratchError),
}

impl From<GgufSemanticError> for ScratchTokenizerError {
    fn from(error: GgufSemanticError) -> Self {
        Self::Semantic(error)
    }
}

impl From<TokenizerInputError> for ScratchTokenizerError {
    fn from(error: TokenizerInputError) -> Self {
        Self::Semantic(tokenizer_input_error(error))
    }
}

/// `build_captured_tokenizer`'s audits and construction, with the BPE merge
/// scratch admitted on `raw` for the duration of the BPE build.
pub(super) fn build_captured_tokenizer_with_merge_scratch(
    snapshot: &GgufCompanionSnapshot,
    directory: &Path,
    vocab: usize,
    style: ChatStyle,
    bos_token: &str,
    raw: &GgufRawSourceBudget,
) -> Result<(BasalTokenizer, u32), GgufMergeScratchReadError> {
    let scratch = MergeScratch { raw };
    build_captured_tokenizer_with(snapshot, directory, vocab, style, bos_token, |original, path, style| {
        BasalTokenizer::from_json_value_with(original, path, style, |text| {
            tokenizers::Tokenizer::from_str_with_merge_storage(text, &scratch).map_err(|error| match error {
                MergeBuildError::Storage(source) => {
                    ScratchTokenizerError::Scratch(GgufMergeScratchError::Admission { source })
                }
                MergeBuildError::BufferLength { requested, actual } => {
                    ScratchTokenizerError::Scratch(GgufMergeScratchError::ProviderLength { requested, actual })
                }
                MergeBuildError::Backend(source) => {
                    TokenizerInputError::Backend { path: path.to_path_buf(), source }.into()
                }
            })
        })
    })
    .map_err(|error| match error {
        ScratchTokenizerError::Semantic(source) => {
            GgufMergeScratchReadError::Existing { source: GgufImmutableSemanticError::Tokenizer { source } }
        }
        ScratchTokenizerError::Scratch(source) => GgufMergeScratchReadError::Scratch { source },
    })
}
