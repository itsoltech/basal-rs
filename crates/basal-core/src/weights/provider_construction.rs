//! Opt-in admission of the tokenizer BPE builder's merge scratch buffer and,
//! on a second opt-in route, of the built BPE's merge hash-table backing. A
//! third opt-in route also admits the retained compiled Oniguruma backing of
//! decoder-tree regexes.
//!
//! Only that temporary buffer, that table's backing blocks and those regex
//! backing blocks are admitted. Tokenizer JSON values, serde helper storage,
//! the table's entire `Arc` allocation (control block, map header, allocator
//! and hasher), vocabulary maps and strings, added vocabulary, other native
//! regex state and caches remain unadmitted, as on the original route.

use std::path::Path;

use tokenizers::utils::merge_storage::{MergeBuildError, MergeStorage, MergeTableAccount, TableAccount, TableRefusal};

use super::immutable_semantics::GgufImmutableSemanticError;
use super::raw_source::{LedgerRefusal, RawCharge, RawLedger};
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

/// Typed refusal of a BPE merge-table backing block. No input text is retained.
#[derive(Debug, thiserror::Error)]
pub enum GgufMergeTableError {
    #[error("BPE merge table admission failed")]
    Admission {
        #[source]
        source: GgufRawSourceError,
    },
    #[error("BPE merge table capacity overflows")]
    CapacityOverflow,
}

/// Failure of `GgufImmutableCompanionSemantics::read_with_bpe_merge_table_budget`.
#[derive(Debug, thiserror::Error)]
pub enum GgufMergeTableReadError {
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
    #[error("GGUF tokenizer BPE merge table was refused")]
    Table {
        #[source]
        source: GgufMergeTableError,
    },
}

/// The ledger is also the merge table's account: the table charges the same
/// held/limit as every other charge of this allowance, with the same policy.
impl TableAccount for RawLedger {
    fn try_acquire(&self, bytes: usize) -> Result<(), TableRefusal> {
        RawLedger::try_acquire(self, bytes).map_err(|refusal| match refusal {
            LedgerRefusal::Busy => TableRefusal::Busy,
            LedgerRefusal::Poisoned => TableRefusal::Closed,
            LedgerRefusal::Overflow => TableRefusal::Overflow,
            LedgerRefusal::Limit { requested, held, limit } => TableRefusal::Limit { requested, held, limit },
        })
    }

    fn release(&self, bytes: usize) {
        RawLedger::release(self, bytes);
    }
}

/// Per-build scratch and merge-table storage on the declaration budget's raw ledger.
struct MergeAdmission<'a> {
    raw: &'a GgufRawSourceBudget,
}

/// Private refusal of either admitted BPE merge storage.
#[derive(Debug, thiserror::Error)]
enum MergeAdmissionError {
    #[error("BPE merge scratch admission failed")]
    Scratch(#[source] GgufRawSourceError),
    #[error("BPE merge table admission failed")]
    Table(#[source] TableRefusal),
}

impl MergeStorage for MergeAdmission<'_> {
    type Buffer = MergeScratchBuffer;
    type Error = MergeAdmissionError;

    /// The scratch buffer exactly as on the scratch-only route.
    fn try_zeroed(&self, length: usize) -> Result<MergeScratchBuffer, MergeAdmissionError> {
        MergeScratch { raw: self.raw }.try_zeroed(length).map_err(MergeAdmissionError::Scratch)
    }

    fn merge_table(&self) -> Option<MergeTableAccount<MergeAdmissionError>> {
        Some(MergeTableAccount { account: self.raw.table_account(), refused: MergeAdmissionError::Table })
    }
}

/// The ledger cause of a table-account refusal; `None` for a capacity overflow.
fn ledger_cause(refusal: TableRefusal) -> Option<GgufRawSourceError> {
    Some(match refusal {
        TableRefusal::Busy => GgufRawSourceError::Busy,
        TableRefusal::Closed => GgufRawSourceError::Poisoned,
        TableRefusal::Overflow => GgufRawSourceError::Overflow,
        TableRefusal::Limit { requested, held, limit } => GgufRawSourceError::Limit { requested, held, limit },
        TableRefusal::Allocation { bytes } => GgufRawSourceError::Allocation { bytes },
        TableRefusal::CapacityOverflow => return None,
    })
}

fn table_error(refusal: TableRefusal) -> GgufMergeTableError {
    match ledger_cause(refusal) {
        Some(source) => GgufMergeTableError::Admission { source },
        None => GgufMergeTableError::CapacityOverflow,
    }
}

/// Private union of the existing semantic channel and an opt-in route's refusals.
enum AdmittedTokenizerError<R> {
    Semantic(GgufSemanticError),
    Refused(R),
}

impl<R> From<GgufSemanticError> for AdmittedTokenizerError<R> {
    fn from(error: GgufSemanticError) -> Self {
        Self::Semantic(error)
    }
}

impl<R> From<TokenizerInputError> for AdmittedTokenizerError<R> {
    fn from(error: TokenizerInputError) -> Self {
        Self::Semantic(tokenizer_input_error(error))
    }
}

/// `build_captured_tokenizer_with_merge_scratch`, and the built BPE's merge
/// table is admitted on `raw` block by block while the returned tokenizer, or
/// any clone of it, holds that table.
pub(super) fn build_captured_tokenizer_with_merge_table(
    snapshot: &GgufCompanionSnapshot,
    directory: &Path,
    vocab: usize,
    style: ChatStyle,
    bos_token: &str,
    raw: &GgufRawSourceBudget,
) -> Result<(BasalTokenizer, u32), GgufMergeTableReadError> {
    let storage = MergeAdmission { raw };
    build_captured_tokenizer_with(snapshot, directory, vocab, style, bos_token, |original, path, style| {
        BasalTokenizer::from_json_value_with(original, path, style, |text| {
            tokenizers::Tokenizer::from_str_with_merge_storage(text, &storage).map_err(|error| match error {
                MergeBuildError::Storage(MergeAdmissionError::Scratch(source)) => {
                    let source = GgufMergeScratchError::Admission { source };
                    AdmittedTokenizerError::Refused(GgufMergeTableReadError::Scratch { source })
                }
                MergeBuildError::Storage(MergeAdmissionError::Table(refusal)) => {
                    AdmittedTokenizerError::Refused(GgufMergeTableReadError::Table { source: table_error(refusal) })
                }
                MergeBuildError::BufferLength { requested, actual } => {
                    let source = GgufMergeScratchError::ProviderLength { requested, actual };
                    AdmittedTokenizerError::Refused(GgufMergeTableReadError::Scratch { source })
                }
                MergeBuildError::Backend(source) => {
                    TokenizerInputError::Backend { path: path.to_path_buf(), source }.into()
                }
            })
        })
    })
    .map_err(|error| match error {
        AdmittedTokenizerError::Semantic(source) => {
            GgufMergeTableReadError::Existing { source: GgufImmutableSemanticError::Tokenizer { source } }
        }
        AdmittedTokenizerError::Refused(error) => error,
    })
}

/// Typed refusal of a decoder-tree regex backing block. No input text is retained.
#[derive(Debug, thiserror::Error)]
pub enum GgufDecoderRegexError {
    #[error("tokenizer decoder regex admission failed")]
    Admission {
        #[source]
        source: GgufRawSourceError,
    },
    #[error("tokenizer decoder regex backing size overflows")]
    CapacityOverflow,
}

/// Failure of `GgufImmutableCompanionSemantics::read_with_decoder_regex_budget`.
#[derive(Debug, thiserror::Error)]
pub enum GgufDecoderRegexReadError {
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
    #[error("GGUF tokenizer BPE merge table was refused")]
    Table {
        #[source]
        source: GgufMergeTableError,
    },
    #[error("GGUF tokenizer decoder regex was refused")]
    DecoderRegex {
        #[source]
        source: GgufDecoderRegexError,
    },
}

/// Merge scratch, merge table and decoder-regex backing on the declaration
/// budget's raw ledger.
struct DecoderRegexAdmission<'a> {
    merge: MergeAdmission<'a>,
}

/// Private refusal of any admitted storage of the decoder-regex route.
#[derive(Debug, thiserror::Error)]
enum DecoderRegexAdmissionError {
    #[error(transparent)]
    Merge(MergeAdmissionError),
    #[error("tokenizer decoder regex admission failed")]
    Regex(#[source] TableRefusal),
}

impl MergeStorage for DecoderRegexAdmission<'_> {
    type Buffer = MergeScratchBuffer;
    type Error = DecoderRegexAdmissionError;

    fn try_zeroed(&self, length: usize) -> Result<MergeScratchBuffer, DecoderRegexAdmissionError> {
        self.merge.try_zeroed(length).map_err(DecoderRegexAdmissionError::Merge)
    }

    fn merge_table(&self) -> Option<MergeTableAccount<DecoderRegexAdmissionError>> {
        Some(MergeTableAccount {
            account: self.merge.raw.table_account(),
            refused: |refusal| DecoderRegexAdmissionError::Merge(MergeAdmissionError::Table(refusal)),
        })
    }

    fn decoder_regex(&self) -> Option<MergeTableAccount<DecoderRegexAdmissionError>> {
        Some(MergeTableAccount { account: self.merge.raw.table_account(), refused: DecoderRegexAdmissionError::Regex })
    }
}

fn decoder_regex_error(refusal: TableRefusal) -> GgufDecoderRegexError {
    match ledger_cause(refusal) {
        Some(source) => GgufDecoderRegexError::Admission { source },
        None => GgufDecoderRegexError::CapacityOverflow,
    }
}

/// `build_captured_tokenizer_with_merge_table`, and the retained compiled
/// Oniguruma backing of every `Replace` regex in the tokenizer's decoder tree
/// is admitted on `raw` block by block while the returned tokenizer, or any
/// clone of it, holds that regex.
pub(super) fn build_captured_tokenizer_with_decoder_regex(
    snapshot: &GgufCompanionSnapshot,
    directory: &Path,
    vocab: usize,
    style: ChatStyle,
    bos_token: &str,
    raw: &GgufRawSourceBudget,
) -> Result<(BasalTokenizer, u32), GgufDecoderRegexReadError> {
    let storage = DecoderRegexAdmission { merge: MergeAdmission { raw } };
    build_captured_tokenizer_with(snapshot, directory, vocab, style, bos_token, |original, path, style| {
        BasalTokenizer::from_json_value_with(original, path, style, |text| {
            tokenizers::Tokenizer::from_str_with_merge_storage(text, &storage).map_err(|error| match error {
                MergeBuildError::Storage(DecoderRegexAdmissionError::Merge(MergeAdmissionError::Scratch(source))) => {
                    let source = GgufMergeScratchError::Admission { source };
                    AdmittedTokenizerError::Refused(GgufDecoderRegexReadError::Scratch { source })
                }
                MergeBuildError::Storage(DecoderRegexAdmissionError::Merge(MergeAdmissionError::Table(refusal))) => {
                    AdmittedTokenizerError::Refused(GgufDecoderRegexReadError::Table { source: table_error(refusal) })
                }
                MergeBuildError::Storage(DecoderRegexAdmissionError::Regex(refusal)) => {
                    let source = decoder_regex_error(refusal);
                    AdmittedTokenizerError::Refused(GgufDecoderRegexReadError::DecoderRegex { source })
                }
                MergeBuildError::BufferLength { requested, actual } => {
                    let source = GgufMergeScratchError::ProviderLength { requested, actual };
                    AdmittedTokenizerError::Refused(GgufDecoderRegexReadError::Scratch { source })
                }
                MergeBuildError::Backend(source) => {
                    TokenizerInputError::Backend { path: path.to_path_buf(), source }.into()
                }
            })
        })
    })
    .map_err(|error| match error {
        AdmittedTokenizerError::Semantic(source) => {
            GgufDecoderRegexReadError::Existing { source: GgufImmutableSemanticError::Tokenizer { source } }
        }
        AdmittedTokenizerError::Refused(error) => error,
    })
}
