//! Explicit, per-call storage for the BPE builder's merge scratch buffer and,
//! optionally, accounts for the built model's merge table and reverse
//! vocabulary table backing and for the compiled backing of decoder-tree
//! regexes.
//!
//! Local basal-rs addition. The legacy `BpeBuilder::build` and `Deserialize`
//! routes keep their own `vec![0; max_len]` allocation and global merge table.
//! Only the typed refusal moves without new formatting or boxing; JSON parsing,
//! syntax-only skipping and serde diagnostics keep their own, unaccounted
//! allocations.

use std::sync::Arc;

/// Supplies the scratch buffer used while a BPE model validates its merges.
///
/// The builder requests exactly the byte length of the longest vocabulary
/// token once per build. It takes one mutable slice from the returned buffer,
/// requires that slice to have exactly the requested length, and uses that same
/// slice for every merge write and read. The buffer is dropped when the build
/// returns; it is not retained by the model.
pub trait MergeStorage {
    /// Owner of the scratch bytes.
    type Buffer: AsMut<[u8]>;
    /// Typed refusal. It is returned unchanged, never boxed or formatted.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Returns a buffer of `length` bytes, or a typed refusal.
    fn try_zeroed(&self, length: usize) -> Result<Self::Buffer, Self::Error>;

    /// The account that admits the built BPE's merge hash-table backing.
    ///
    /// `None` (the default) keeps the merge table on the global allocator, as
    /// the legacy build does. The account is requested once before the build
    /// starts. With `Some`, every table block is then admitted before allocation
    /// and released after deallocation, while the returned model or any of its
    /// clones holds the table.
    fn merge_table(&self) -> Option<MergeTableAccount<Self::Error>> {
        None
    }

    /// The account that admits the built BPE's reverse vocabulary (ID to
    /// token) hash-table backing.
    ///
    /// `None` (the default) keeps the reverse vocabulary on the global
    /// allocator, as the legacy build does. The account is requested once
    /// before the build starts, after [`merge_table`](Self::merge_table). With
    /// `Some`, the table's single capacity request, the vocabulary length, is
    /// admitted before allocation and before any token is copied, and released
    /// after deallocation, while the returned model or any of its clones holds
    /// the table. The copied token strings stay on the global allocator.
    fn reverse_vocab_table(&self) -> Option<MergeTableAccount<Self::Error>> {
        None
    }

    /// The account that admits the compiled backing of decoder-tree regexes.
    ///
    /// Requested when the tokenizer's `decoder` entry is reached. `None` (the
    /// default) builds the decoder exactly as the legacy route does. With
    /// `Some`, the retained compiled Oniguruma backing of every `Replace` in
    /// the decoder tree is admitted before allocation and released after it is
    /// freed, while the decoder or any of its clones holds the regex. Normalizer
    /// and pre-tokenizer regexes stay unadmitted. Without the `onig` feature
    /// the account is not requested.
    fn decoder_regex(&self) -> Option<MergeTableAccount<Self::Error>> {
        None
    }
}

/// Account charged for each charged block: a merge-table or reverse
/// vocabulary table block or, through the decoder-regex account, a compiled
/// regex backing block.
pub trait TableAccount: Send + Sync + std::panic::RefUnwindSafe + std::panic::UnwindSafe {
    /// Admits `bytes` before their allocation. A refusal leaves the account
    /// unchanged.
    fn try_acquire(&self, bytes: usize) -> Result<(), TableRefusal>;

    /// Returns `bytes` admitted earlier, after their block was freed or its
    /// allocation failed. It may run on any thread that drops the last owner.
    /// It must not panic, nor allocate through a charged table.
    fn release(&self, bytes: usize);
}

/// Fixed refusal of one charged block; it retains no input. The messages name
/// the merge table for every charged table; the storage's per-account
/// `refused` function identifies which table or regex was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TableRefusal {
    /// The account is in use by another admission.
    #[error("merge table account is busy")]
    Busy,
    /// The account is poisoned or closed.
    #[error("merge table account is poisoned or closed")]
    Closed,
    /// The account's charge arithmetic overflows.
    #[error("merge table charge arithmetic overflows")]
    Overflow,
    /// The block exceeds the account's remaining allowance.
    #[error(
        "merge table block of {requested} bytes exceeds allowance (held {held}, limit {limit})"
    )]
    Limit {
        requested: usize,
        held: usize,
        limit: usize,
    },
    /// The admitted block could not be allocated; its charge was released.
    #[error("merge table allocation of {bytes} bytes failed")]
    Allocation { bytes: usize },
    /// The table's capacity computation overflows before any request.
    #[error("merge table capacity overflows")]
    CapacityOverflow,
}

/// The caller's account for one charged table of one build (the merge table
/// or the reverse vocabulary) or for the decoder regexes.
pub struct MergeTableAccount<E> {
    /// Charged for every table block; retained by the table until it is freed.
    pub account: Arc<dyn TableAccount>,
    /// Converts a table refusal into the storage's typed refusal.
    pub refused: fn(TableRefusal) -> E,
}

/// The [`MergeStorage::decoder_regex`] account as an Oniguruma regex account.
/// Every admitted regex retains it until its last owner drops the regex.
#[cfg(feature = "onig")]
pub(crate) struct RegexCharge(pub(crate) Arc<dyn TableAccount>);

#[cfg(feature = "onig")]
impl onig::RegexAccount for RegexCharge {
    type Refusal = TableRefusal;

    fn try_acquire(&self, bytes: usize) -> Result<(), TableRefusal> {
        self.0.try_acquire(bytes)
    }

    fn release(&self, bytes: usize) {
        self.0.release(bytes);
    }
}

/// Failure of an admitted decoder regex.
#[cfg(feature = "onig")]
pub(crate) enum AdmittedRegexError {
    /// Typed refusal of a compiled-backing block, never boxed or formatted.
    Refused(TableRefusal),
    /// The legacy compile error, boxed as the legacy constructor boxes it.
    Backend(Box<dyn std::error::Error + Send + Sync>),
}

/// Failure of a build that uses explicit [`MergeStorage`].
#[derive(Debug, thiserror::Error)]
pub enum MergeBuildError<E> {
    /// The storage refused the scratch buffer or, when it supplies a
    /// [`MergeStorage::merge_table`], [`MergeStorage::reverse_vocab_table`]
    /// or [`MergeStorage::decoder_regex`] account, a merge-table, reverse
    /// vocabulary table or decoder-regex block. The message is unchanged; the
    /// source tells them apart.
    #[error("BPE merge scratch storage was refused")]
    Storage(#[source] E),
    /// The storage returned a slice of another length.
    #[error("BPE merge scratch buffer has {actual} bytes, but {requested} were requested")]
    BufferLength { requested: usize, actual: usize },
    /// Any existing parsing or model construction error, unchanged.
    #[error("tokenizer backend construction failed")]
    Backend(#[from] crate::Error),
}
