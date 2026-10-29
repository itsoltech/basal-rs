//! Explicit, per-call storage for the BPE builder's merge scratch buffer.
//!
//! Local basal-rs addition. The legacy `BpeBuilder::build` and `Deserialize`
//! routes keep their own `vec![0; max_len]` allocation. Only the typed refusal
//! moves without new formatting or boxing; JSON parsing, syntax-only skipping
//! and serde diagnostics keep their own, unaccounted allocations.

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
}

/// Failure of a build that uses explicit [`MergeStorage`].
#[derive(Debug, thiserror::Error)]
pub enum MergeBuildError<E> {
    /// The storage refused the scratch buffer.
    #[error("BPE merge scratch storage was refused")]
    Storage(#[source] E),
    /// The storage returned a slice of another length.
    #[error("BPE merge scratch buffer has {actual} bytes, but {requested} were requested")]
    BufferLength { requested: usize, actual: usize },
    /// Any existing parsing or model construction error, unchanged.
    #[error("tokenizer backend construction failed")]
    Backend(#[from] crate::Error),
}
