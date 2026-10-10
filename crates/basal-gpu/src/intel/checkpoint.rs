//! Read-only safetensors access through positioned reads. The checkpoint is not mapped: changing the file while a
//! model is loaded can produce wrong values or an I/O error, but not undefined behaviour.
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use safetensors::tensor::Metadata;

/// Upper bound of the JSON header, as in the `safetensors` crate.
const MAX_HEADER_BYTES: u64 = 100_000_000;
/// Even, so that a chunk never splits a BF16 value.
const CHUNK_BYTES: usize = 8 << 20;

pub(super) struct Checkpoint {
    file: File,
    metadata: Metadata,
    /// File offset of the data section.
    data_start: u64,
}

/// A validated BF16 tensor: absolute file offset and size in bytes.
#[derive(Clone, Copy)]
pub(super) struct Bf16 {
    offset: u64,
    bytes: usize,
}

impl Checkpoint {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let file_len = file.metadata()?.len();
        let mut n = [0u8; 8];
        file.read_exact_at(&mut n, 0).context("reading safetensors header size")?;
        let n = u64::from_le_bytes(n);
        ensure!(n <= MAX_HEADER_BYTES && 8 + n <= file_len, "invalid safetensors header size {n}");
        let mut header = vec![0u8; usize::try_from(n)?];
        file.read_exact_at(&mut header, 8).context("reading safetensors header")?;
        // Deserialisation validates contiguous, non-overlapping offsets and their sizes against shape and dtype.
        let metadata: Metadata = serde_json::from_slice(&header).context("parsing safetensors header")?;
        let data_start = 8 + n;
        ensure!(metadata.data_len() as u64 == file_len - data_start, "safetensors data size disagrees with the file");
        Ok(Self { file, metadata, data_start })
    }

    /// The BF16 tensor `name` with exactly `shape`.
    pub fn bf16(&self, name: &str, shape: &[usize]) -> Result<Bf16> {
        let info = self.metadata.info(name).with_context(|| format!("checkpoint tensor {name} missing"))?;
        ensure!(info.shape == shape, "{name}: shape {:?}, expected {:?}", info.shape, shape);
        ensure!(info.dtype == safetensors::Dtype::BF16, "{name}: dtype {:?}, expected BF16", info.dtype);
        let (start, end) = info.data_offsets;
        Ok(Bf16 { offset: self.data_start + start as u64, bytes: end - start })
    }

    pub fn names(&self) -> Vec<String> {
        self.metadata.offset_keys()
    }

    /// Pass the whole tensor to `sink` in chunks of complete BF16 values, reusing `buffer`.
    pub fn read_chunks(&self, tensor: Bf16, buffer: &mut Vec<u8>, mut sink: impl FnMut(&[u8])) -> Result<()> {
        buffer.resize(CHUNK_BYTES.min(tensor.bytes), 0);
        let mut done = 0;
        while done < tensor.bytes {
            let chunk = &mut buffer[..CHUNK_BYTES.min(tensor.bytes - done)];
            self.file.read_exact_at(chunk, tensor.offset + done as u64).context("reading checkpoint tensor")?;
            sink(chunk);
            done += chunk.len();
        }
        Ok(())
    }

    /// Rows `ids` of a BF16 matrix whose rows have `row_bytes` bytes, concatenated.
    pub fn rows(&self, matrix: Bf16, row_bytes: usize, ids: &[u32]) -> Result<Vec<u8>> {
        let rows = matrix.bytes / row_bytes;
        let mut out = vec![0u8; ids.len() * row_bytes];
        for (&id, row) in ids.iter().zip(out.chunks_exact_mut(row_bytes)) {
            ensure!((id as usize) < rows, "token {id} outside vocabulary");
            self.file
                .read_exact_at(row, matrix.offset + (id as usize * row_bytes) as u64)
                .context("reading checkpoint rows")?;
        }
        Ok(out)
    }
}
