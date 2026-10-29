//! Partial, host-only inspection of one explicitly selected dense Llama GGUF.
//!
//! Separate consuming phases check the bounded header/inventory and then the
//! full-file digest, finite payload and requested conversion policy. Neither
//! grants materialization, inference or memory-admission authority.

use std::collections::TryReserveError;
use std::error::Error;
use std::fmt::{self, Write as _};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};

use basal_core::architecture::Architecture;
use basal_core::manifest::LlamaConfig;
use basal_core::weights::{GgufCompanionSemantics, GgufOuttype};
use candle_core::quantized::gguf_file::{Content, ReadLimits, Value, ValueType};
use candle_core::quantized::GgmlDType;
use sha2::{Digest, Sha256};

use crate::Precision;

const MAX_FILE_BYTES: u64 = 64 << 30;
const MAX_TENSOR_BYTES: u64 = 8 << 30;
const MAX_TENSORS: u64 = 8195;
const MAX_SPAN_SCRATCH: u64 = 1 << 20;
const LOOKUP_BYTES: usize = 128;
const SCAN_BUFFER_BYTES: u64 = 1_048_576;
const TWO_COPY_STAGING_LIMIT: u64 = 2_147_483_648;

/// Explicit acknowledgement of a caller-arranged trusted-local no-writer contract.
///
/// This token acquires no lock or external guard and proves no immutability.
/// It is not a Rust safety prerequisite: safe file I/O remains safe if the
/// assertion is false. Detectable source changes are refused, but racing or
/// restored writes can evade descriptor metadata observations.
#[must_use]
pub struct CallerImmutableLease {
    _private: (),
}

impl CallerImmutableLease {
    /// Acknowledge exclusive no-writer control of the selected inode, arranged
    /// before inspection and maintained through every future last read and
    /// publication. The caller must retain any actual external guard itself.
    ///
    /// If this cannot be fulfilled, do not inspect: a future integration must
    /// arrange an immutable snapshot or refuse the source. No guard is acquired
    /// here, and inspection never fabricates this acknowledgement.
    pub fn acknowledge_trusted_local_contract() -> Self {
        Self { _private: () }
    }
}

/// An owning, partial header/inventory result, with no payload/digest approval.
///
/// All source owners remain private and together. There is no file/map escape,
/// serialization, cloning or materialization API. Future phases must consume
/// these same owners rather than reopen paths from the manifest.
#[must_use]
pub struct GgufHeaderInspection {
    // Retained together for consuming phases, without exposing source handles.
    _file: File,
    _content: Content,
    _stamp: SourceStamp,
    _lease: CallerImmutableLease,
    companions: GgufCompanionSemantics,
    tensor_count: usize,
    source_tensor_bytes: u64,
}

impl GgufHeaderInspection {
    /// Inspect the header/inventory of exactly one transferred regular file.
    ///
    /// Supports little-endian 64-bit Linux/macOS hosts, independently of GPU
    /// features. Seeks to zero, checks literal GGUF/v3, parses once under finite
    /// quotas, crosschecks companions and dense inventory, then compares same-fd
    /// stamps. No companion parsing, tokenizer construction, payload read, digest,
    /// device selection or filesystem discovery occurs. Failure drops all owners.
    pub fn inspect(
        mut file: File,
        companions: GgufCompanionSemantics,
        lease: CallerImmutableLease,
    ) -> Result<Self, GgufInspectionError> {
        if !cfg!(all(
            unix,
            target_endian = "little",
            target_pointer_width = "64",
            any(target_os = "linux", target_os = "macos")
        )) {
            return Err(GgufInspectionError::UnsupportedHost);
        }
        let stamp = SourceStamp::capture(&file, "before inspection")?;
        if stamp.file_type != 0o100000 {
            return Err(GgufInspectionError::NonRegular);
        }
        if stamp.len < 8 {
            return Err(GgufInspectionError::ResourceLimit {
                resource: "minimum file bytes",
                actual: stamp.len,
                limit: 8,
            });
        }
        limit("file bytes", stamp.len, MAX_FILE_BYTES)?;
        let cfg = match &companions.manifest().config {
            Architecture::Llama(cfg) => cfg,
            _ => return Err(GgufInspectionError::CompanionConfig { field: "architecture" }),
        };
        let count = add(3, mul(if cfg.bias { 16 } else { 9 }, integer(cfg.num_layers)?)?)?;
        limit("tensor count", count, MAX_TENSORS)?;
        let tensor_count = usize::try_from(count).map_err(|_| overflow("tensor count to usize"))?;

        file.seek(SeekFrom::Start(0)).map_err(|source| io_error("seek signature", source))?;
        let mut signature = [0; 8];
        file.read_exact(&mut signature).map_err(|source| io_error("read signature", source))?;
        if &signature[..4] != b"GGUF" {
            return Err(GgufInspectionError::Magic);
        }
        let version = u32::from_le_bytes([signature[4], signature[5], signature[6], signature[7]]);
        if version != 3 {
            return Err(GgufInspectionError::Version { actual: version });
        }
        file.seek(SeekFrom::Start(0)).map_err(|source| io_error("seek header", source))?;
        let limits = ReadLimits {
            max_header_bytes: 64 << 20,
            max_metadata_entries: 4096,
            max_tensors: count,
            max_string_bytes: 1 << 20,
            max_total_string_bytes: 32 << 20,
            max_total_values: 1 << 20,
            max_nesting: 2,
            max_rank: 2,
            max_allocation_bytes: 256 << 20,
        };
        let content =
            Content::read_with_limits(&mut file, &limits).map_err(|source| GgufInspectionError::Parse { source })?;
        // Candle leaves the cursor at inventory end, before alignment padding.
        let header_end = file.stream_position().map_err(|source| io_error("header position", source))?;
        let outtype = companions.snapshot().declaration().outtype();
        check_metadata(&content, cfg, outtype, count)?;
        let source_tensor_bytes = check_inventory(&content, cfg, outtype, count, header_end, stamp.len)?;
        stamp.compare(&SourceStamp::capture(&file, "after inspection")?)?;
        Ok(Self {
            _file: file,
            _content: content,
            _stamp: stamp,
            _lease: lease,
            companions,
            tensor_count,
            source_tensor_bytes,
        })
    }

    /// Borrow the actual retained companions, without recapture or reconstruction.
    pub fn companions(&self) -> &GgufCompanionSemantics {
        &self.companions
    }

    /// Exact checked number of required inventory entries.
    pub fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    /// Checked dense source bytes, excluding header, gaps and trailing padding.
    /// This is neither resident memory nor a payload/digest verification.
    pub fn source_tensor_bytes(&self) -> u64 {
        self.source_tensor_bytes
    }

    /// Consume this exact owner to check the full-file digest and dense scalars.
    ///
    /// Checks resident and provisional two-copy costs before reserving 1 MiB
    /// byte scratch plus a 128-byte lookup string. Allowances are comparisons,
    /// not held reservations. Zero is not unlimited; equality is accepted.
    /// The caller's no-writer obligation continues through every later read
    /// and publication. Metadata and digest observations acquire no lease.
    pub fn inspect_payload(
        mut self,
        requested: Precision,
        allowances: GgufInspectionAllowances,
    ) -> Result<GgufPayloadInspection, GgufInspectionError> {
        self._stamp.compare(&SourceStamp::capture(&self._file, "payload inspection entry")?)?;
        let cfg = match &self.companions.manifest().config {
            Architecture::Llama(cfg) => cfg,
            _ => return Err(GgufInspectionError::CompanionConfig { field: "architecture" }),
        };
        let outtype = self.companions.snapshot().declaration().outtype();
        let costs = policy_costs(cfg, outtype, requested)?;
        check_source_bytes(costs.source_tensor_bytes, self.source_tensor_bytes)?;
        // Establish representability before any scratch allocation. Each scan
        // chunk is at most R, so its later bounded cast also fits usize.
        for bytes in [
            costs.source_tensor_bytes,
            costs.resident_weight_bytes,
            costs.max_assembly_group_bytes,
            costs.two_copy_staging_bytes,
        ] {
            usize::try_from(bytes).map_err(|_| overflow("policy bytes to usize"))?;
        }
        let capacity = usize::try_from(SCAN_BUFFER_BYTES).map_err(|_| overflow("scan capacity"))?;
        if costs.resident_weight_bytes > allowances.remaining_resident_bytes {
            return Err(GgufInspectionError::ResidentBudget {
                required: costs.resident_weight_bytes,
                remaining: allowances.remaining_resident_bytes,
            });
        }
        let effective_limit = TWO_COPY_STAGING_LIMIT.min(allowances.remaining_staging_bytes);
        if costs.two_copy_staging_bytes > effective_limit {
            return Err(GgufInspectionError::StagingBudget { required: costs.two_copy_staging_bytes, effective_limit });
        }
        let mut scratch = Vec::new();
        scratch.try_reserve_exact(capacity).map_err(|source| GgufInspectionError::Allocation { source })?;
        scratch.resize(capacity, 0);
        let mut name = String::new();
        name.try_reserve_exact(LOOKUP_BYTES).map_err(|source| GgufInspectionError::Allocation { source })?;

        self._stamp.compare(&SourceStamp::capture(&self._file, "before digest")?)?;
        let observed_sha256 = hash_source(&mut self._file, &self._stamp, &mut scratch)?;
        let declared = self.companions.snapshot().declaration().declared_gguf_sha256();
        let expected = declared.as_bytes();
        if &observed_sha256 != expected {
            return Err(GgufInspectionError::GgufDigestMismatch { expected: *expected, actual: observed_sha256 });
        }
        self._stamp.compare(&SourceStamp::capture(&self._file, "before payload scan")?)?;
        let mut scanned_bytes = 0;
        visit_roles(cfg, outtype, &mut name, |name, shape, dtype, width, head| {
            let info = self
                ._content
                .tensor_infos
                .get(name)
                .ok_or_else(|| GgufInspectionError::MissingTensor { tensor: bounded(name) })?;
            let bytes = mul(shape.iter().try_fold(1, |n, &dim| mul(n, dim))?, width)?;
            let start = add(self._content.tensor_data_offset, info.offset)?;
            let end = add(start, bytes)?;
            if start < self._content.tensor_data_offset || start >= end || end > self._stamp.len {
                return Err(GgufInspectionError::TensorSpan { tensor: bounded(name), start, end });
            }
            self._file.seek(SeekFrom::Start(start)).map_err(|source| io_error("seek payload span", source))?;
            scan_tensor(&mut self._file, &mut scratch, name, bytes, dtype, requested, head)?;
            scanned_bytes = add(scanned_bytes, bytes)?;
            Ok(())
        })?;
        check_source_bytes(costs.source_tensor_bytes, scanned_bytes)?;
        self._file.seek(SeekFrom::Start(self._stamp.len)).map_err(|source| io_error("seek payload EOF", source))?;
        probe_eof(&mut self._file, "probe after payload scan", "growth after payload scan")?;
        self._stamp.compare(&SourceStamp::capture(&self._file, "after payload scan")?)?;
        self._stamp.compare(&SourceStamp::capture(&self._file, "payload inspection success")?)?;
        drop(scratch);
        drop(name);
        Ok(GgufPayloadInspection { header: self, requested, observed_sha256, costs })
    }
}

/// Required remaining byte allowances for arithmetic comparisons only.
/// Neither field represents a held reservation; zero is not unlimited.
pub struct GgufInspectionAllowances {
    pub remaining_resident_bytes: u64,
    pub remaining_staging_bytes: u64,
}

/// Owning partial digest/payload result, with no allocation authority.
///
/// Retains the entire exact header owner and freezes the requested software
/// scalar policy. Future materialization must consume this owner, use that
/// policy without substituting precision or bulk casts, stamp its reads and
/// rehash the same descriptor before publication. This observation proves
/// neither immutable bytes nor source authenticity or durable identity.
#[must_use]
pub struct GgufPayloadInspection {
    header: GgufHeaderInspection,
    requested: Precision,
    observed_sha256: [u8; 32],
    costs: GgufPolicyCosts,
}

impl GgufPayloadInspection {
    /// Borrow the actual retained companion and calibration semantics.
    pub fn companions(&self) -> &GgufCompanionSemantics {
        self.header.companions()
    }
    /// Exact number of inspected source tensors.
    pub fn tensor_count(&self) -> usize {
        self.header.tensor_count()
    }
    /// Frozen requested storage/conversion policy, distinct from legacy storage.
    pub fn requested_precision(&self) -> Precision {
        self.requested
    }
    /// Observed full original-length SHA-256, equal to the declaration.
    pub fn observed_gguf_sha256(&self) -> &[u8; 32] {
        &self.observed_sha256
    }
    /// Checked costs, without any reservation or materialization authority.
    pub fn costs(&self) -> &GgufPolicyCosts {
        &self.costs
    }
}

/// Dense element-byte costs, excluding allocator/driver and runtime overhead.
pub struct GgufPolicyCosts {
    source_tensor_bytes: u64,
    resident_weight_bytes: u64,
    max_assembly_group_bytes: u64,
    two_copy_staging_bytes: u64,
}

impl GgufPolicyCosts {
    /// Source spans only, excluding header, padding and synthetic zero biases.
    pub fn source_tensor_bytes(&self) -> u64 {
        self.source_tensor_bytes
    }
    /// Final weight bytes, including absent biases as synthetic zeros once.
    /// Excludes compute, KV, RoPE, readout and gather scratch.
    pub fn resident_weight_bytes(&self) -> u64 {
        self.resident_weight_bytes
    }
    /// Largest final assembly group, including grouped and zero biases.
    pub fn max_assembly_group_bytes(&self) -> u64 {
        self.max_assembly_group_bytes
    }
    /// PROVISIONAL engineering plan: twice the largest group plus 1 MiB.
    /// This is not an actual materialization peak, reservation or admission.
    /// CPU staging can overlap three group copies; a future uploader must count
    /// all actual copies and pending/error-retained owners and reacquire held
    /// allowances before allocating a model. Inspection allocates no group.
    pub fn two_copy_staging_bytes(&self) -> u64 {
        self.two_copy_staging_bytes
    }
}

/// Implemented host/header/inventory/digest/payload and policy-cost refusals.
///
/// Top-level Display/Debug bounds key/name/value text and omits nested causes,
/// buffers, JSON, origin and paths. Callers must treat `source()` diagnostics
/// from Candle, I/O and allocation as untrusted; those may contain input text.
#[non_exhaustive]
pub enum GgufInspectionError {
    UnsupportedHost,
    NonRegular,
    SourceChanged { field: &'static str },
    Io { operation: &'static str, source: io::Error },
    Magic,
    Version { actual: u32 },
    Parse { source: candle_core::Error },
    ResourceLimit { resource: &'static str, actual: u64, limit: u64 },
    ArithmeticOverflow { operation: &'static str },
    Allocation { source: TryReserveError },
    CompanionConfig { field: &'static str },
    MissingMetadata { key: &'static str },
    MetadataType { key: String, expected: &'static str, actual: ValueType },
    MetadataValue { key: String, expected: String, actual: String },
    UnsupportedMetadata { key: String },
    SplitUnsupported { key: String },
    TensorCount { expected: u64, actual: u64 },
    MissingTensor { tensor: String },
    TensorType { tensor: String, expected: GgmlDType, actual: GgmlDType },
    TensorShape { tensor: String },
    TensorAlignment { tensor: String },
    TensorSpan { tensor: String, start: u64, end: u64 },
    DataOffset { expected: u64, actual: u64 },
    OverlappingTensors { previous_end: u64, next_start: u64 },
    SourceBytes { expected: u64, actual: u64 },
    GgufDigestMismatch { expected: [u8; 32], actual: [u8; 32] },
    NonFinitePayload { tensor: String, element: u64 },
    NonFiniteConversion { tensor: String, element: u64, requested: Precision },
    ResidentBudget { required: u64, remaining: u64 },
    StagingBudget { required: u64, effective_limit: u64 },
}

// Bound output even for errors constructed directly by an external caller.
struct Text<'a>(&'a str);
impl fmt::Display for Text<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for c in self.0.chars().take(96) {
            for escaped in c.escape_default() {
                f.write_char(escaped)?;
            }
        }
        if self.0.chars().nth(96).is_some() {
            f.write_str("…")?;
        }
        Ok(())
    }
}

impl fmt::Display for GgufInspectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedHost => f.write_str("GGUF inspection requires a little-endian 64-bit Linux/macOS host"),
            Self::NonRegular => f.write_str("GGUF source is not a regular file"),
            Self::SourceChanged { field } => write!(f, "GGUF source changed during inspection: {}", Text(field)),
            Self::Io { operation, .. } => write!(f, "GGUF inspection I/O: {}", Text(operation)),
            Self::Magic => f.write_str("GGUF literal magic mismatch"),
            Self::Version { actual } => write!(f, "GGUF version {actual}; expected 3"),
            Self::Parse { .. } => f.write_str("bounded GGUF header parser refused input"),
            Self::ResourceLimit { resource, actual, limit } => {
                write!(f, "GGUF {}: {actual}, bound {limit}", Text(resource))
            }
            Self::ArithmeticOverflow { operation } => write!(f, "GGUF arithmetic overflow: {}", Text(operation)),
            Self::Allocation { .. } => f.write_str("GGUF inspection scratch reservation failed"),
            Self::CompanionConfig { field } => {
                write!(f, "GGUF companion config cannot be represented: {}", Text(field))
            }
            Self::MissingMetadata { key } => write!(f, "GGUF missing metadata {}", Text(key)),
            Self::MetadataType { key, expected, actual } => {
                write!(f, "GGUF metadata {}: expected {}, actual {actual:?}", Text(key), Text(expected))
            }
            Self::MetadataValue { key, expected, actual } => {
                write!(f, "GGUF metadata {}: expected {}, actual {}", Text(key), Text(expected), Text(actual))
            }
            Self::UnsupportedMetadata { key } => write!(f, "GGUF unsupported metadata {}", Text(key)),
            Self::SplitUnsupported { key } => write!(f, "GGUF unsupported split metadata {}", Text(key)),
            Self::TensorCount { expected, actual } => write!(f, "GGUF tensor count {actual}; expected {expected}"),
            Self::MissingTensor { tensor } => write!(f, "GGUF missing tensor {}", Text(tensor)),
            Self::TensorType { tensor, expected, actual } => {
                write!(f, "GGUF tensor {}: expected {expected:?}, actual {actual:?}", Text(tensor))
            }
            Self::TensorShape { tensor } => {
                write!(f, "GGUF tensor {}: wrong logical rank/shape or nonpositive dimension", Text(tensor))
            }
            Self::TensorAlignment { tensor } => {
                write!(f, "GGUF tensor {}: offset/start is not 32-byte aligned", Text(tensor))
            }
            Self::TensorSpan { tensor, start, end } => {
                write!(f, "GGUF tensor {}: invalid source span [{start}, {end})", Text(tensor))
            }
            Self::DataOffset { expected, actual } => {
                write!(f, "GGUF data base {actual}; expected {expected} within file")
            }
            Self::OverlappingTensors { previous_end, next_start } => {
                write!(f, "GGUF overlapping spans: previous end {previous_end}, next start {next_start}")
            }
            Self::SourceBytes { expected, actual } => {
                write!(f, "GGUF source tensor bytes {actual}; expected {expected}")
            }
            Self::GgufDigestMismatch { expected, actual } => {
                f.write_str("GGUF SHA-256 mismatch: expected ")?;
                for byte in expected {
                    write!(f, "{byte:02x}")?;
                }
                f.write_str(", actual ")?;
                for byte in actual {
                    write!(f, "{byte:02x}")?;
                }
                Ok(())
            }
            Self::NonFinitePayload { tensor, element } => {
                write!(f, "GGUF nonfinite payload in {} at source element {element}", Text(tensor))
            }
            Self::NonFiniteConversion { tensor, element, requested } => {
                write!(f, "GGUF nonfinite {requested:?} conversion in {} at source element {element}", Text(tensor))
            }
            Self::ResidentBudget { required, remaining } => {
                write!(f, "GGUF resident weight bytes require {required}; remaining {remaining}")
            }
            Self::StagingBudget { required, effective_limit } => {
                write!(
                    f,
                    "GGUF provisional two-copy staging plan requires {required}; effective limit {effective_limit}"
                )
            }
        }
    }
}

impl fmt::Debug for GgufInspectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl Error for GgufInspectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Parse { source } => Some(source),
            Self::Allocation { source } => Some(source),
            _ => None,
        }
    }
}

fn io_error(operation: &'static str, source: io::Error) -> GgufInspectionError {
    GgufInspectionError::Io { operation, source }
}
fn overflow(operation: &'static str) -> GgufInspectionError {
    GgufInspectionError::ArithmeticOverflow { operation }
}
fn integer(value: usize) -> Result<u64, GgufInspectionError> {
    u64::try_from(value).map_err(|_| overflow("usize to u64"))
}
fn add(a: u64, b: u64) -> Result<u64, GgufInspectionError> {
    a.checked_add(b).ok_or_else(|| overflow("addition"))
}
fn mul(a: u64, b: u64) -> Result<u64, GgufInspectionError> {
    a.checked_mul(b).ok_or_else(|| overflow("multiplication"))
}
fn limit(resource: &'static str, actual: u64, limit: u64) -> Result<(), GgufInspectionError> {
    if actual > limit {
        return Err(GgufInspectionError::ResourceLimit { resource, actual, limit });
    }
    Ok(())
}
fn bounded(text: &str) -> String {
    text.chars().take(96).collect()
}

fn check_source_bytes(expected: u64, actual: u64) -> Result<(), GgufInspectionError> {
    if actual != expected {
        return Err(GgufInspectionError::SourceBytes { expected, actual });
    }
    Ok(())
}

fn policy_costs(
    cfg: &LlamaConfig,
    outtype: GgufOuttype,
    requested: Precision,
) -> Result<GgufPolicyCosts, GgufInspectionError> {
    let (h, i, v, l) =
        (integer(cfg.hidden)?, integer(cfg.intermediate)?, integer(cfg.vocab)?, integer(cfg.num_layers)?);
    let k = mul(integer(cfg.kv_heads)?, integer(cfg.head_dim)?)?;
    let (_, m) = matrix_type(outtype);
    let head = mul(v, h)?;
    let layer_matrix = add(add(mul(mul(2, h)?, h)?, mul(mul(2, h)?, k)?)?, mul(mul(3, h)?, i)?)?;
    let layer_matrices = mul(l, layer_matrix)?;
    let matrices = add(mul(2, head)?, layer_matrices)?;
    let norms = mul(add(mul(2, l)?, 1)?, h)?;
    let layer_bias = add(add(mul(3, h)?, mul(2, k)?)?, mul(2, i)?)?;
    let biases = mul(l, layer_bias)?;
    let (present_bias, zero_bias) = if cfg.bias { (biases, 0) } else { (0, biases) };
    let source_tensor_bytes = add(mul(m, matrices)?, mul(4, add(norms, present_bias)?)?)?;
    let (t, u, resident_weight_bytes) = match requested {
        Precision::F16 | Precision::Bf16 => {
            let nonhead = add(add(add(head, layer_matrices)?, norms)?, add(present_bias, zero_bias)?)?;
            (2, 2, add(mul(m, head)?, mul(2, nonhead)?)?)
        }
        Precision::F32 => (m, 4, add(source_tensor_bytes, mul(4, zero_bias)?)?),
    };
    let qkv = add(h, mul(2, k)?)?;
    let gate_up = mul(2, i)?;
    // Fixed final groups, not an inventory or a promise about upload order.
    // Norm and bias groups apply even when biases must be synthesized.
    let max_assembly_group_bytes = [
        mul(head, t)?,
        mul(head, m)?,
        mul(h, u)?,
        mul(mul(qkv, h)?, t)?,
        mul(mul(h, h)?, t)?,
        mul(mul(gate_up, h)?, t)?,
        mul(mul(h, i)?, t)?,
        mul(qkv, u)?,
        mul(h, u)?,
        mul(gate_up, u)?,
        mul(h, u)?,
    ]
    .into_iter()
    .fold(0, u64::max);
    let two_copy_staging_bytes = add(mul(2, max_assembly_group_bytes)?, SCAN_BUFFER_BYTES)?;
    Ok(GgufPolicyCosts { source_tensor_bytes, resident_weight_bytes, max_assembly_group_bytes, two_copy_staging_bytes })
}

// Only reads/probes retry Interrupted. Other I/O failures retain their causes
// and return immediately, without a best-effort stamp masking the first error.
fn read_source(file: &mut File, bytes: &mut [u8], operation: &'static str) -> Result<usize, GgufInspectionError> {
    loop {
        match file.read(bytes) {
            Err(source) if source.kind() == io::ErrorKind::Interrupted => continue,
            result => return result.map_err(|source| io_error(operation, source)),
        }
    }
}

fn probe_eof(file: &mut File, operation: &'static str, field: &'static str) -> Result<(), GgufInspectionError> {
    if read_source(file, &mut [0; 1], operation)? != 0 {
        return Err(GgufInspectionError::SourceChanged { field });
    }
    Ok(())
}

fn hash_source(file: &mut File, original: &SourceStamp, scratch: &mut [u8]) -> Result<[u8; 32], GgufInspectionError> {
    file.seek(SeekFrom::Start(0)).map_err(|source| io_error("seek digest", source))?;
    let mut hasher = Sha256::new();
    let mut read_bytes = 0;
    while read_bytes < original.len {
        let remaining = original.len.checked_sub(read_bytes).ok_or_else(|| overflow("digest remaining"))?;
        // Bounded by R, whose usize conversion preceded scratch allocation.
        let chunk = remaining.min(SCAN_BUFFER_BYTES) as usize;
        let n = read_source(file, &mut scratch[..chunk], "read digest")?;
        if n == 0 {
            return Err(GgufInspectionError::SourceChanged { field: "EOF during digest" });
        }
        hasher.update(&scratch[..n]);
        read_bytes = add(read_bytes, integer(n)?)?;
    }
    probe_eof(file, "probe after digest", "growth after digest")?;
    original.compare(&SourceStamp::capture(file, "after digest")?)?;
    Ok(hasher.finalize().into())
}

fn scan_tensor(
    file: &mut File,
    scratch: &mut [u8],
    name: &str,
    bytes: u64,
    dtype: GgmlDType,
    requested: Precision,
    head: bool,
) -> Result<(), GgufInspectionError> {
    // The private role table supplies only these scalar types. Returning a
    // typed refusal also avoids any input-dependent panic if that table changes.
    let width = match dtype {
        GgmlDType::F16 | GgmlDType::BF16 => 2,
        GgmlDType::F32 => 4,
        actual => {
            return Err(GgufInspectionError::TensorType { tensor: bounded(name), expected: GgmlDType::F32, actual });
        }
    };
    let mut remaining = bytes;
    let mut element = 0;
    while remaining != 0 {
        // R and every role byte count are divisible by width. A full chunk is
        // filled before decoding, even if individual OS reads split scalars.
        // The bounded cast fits the prechecked R capacity on supported hosts.
        let chunk = remaining.min(SCAN_BUFFER_BYTES) as usize;
        let mut filled = 0;
        while filled < chunk {
            let n = read_source(file, &mut scratch[filled..chunk], "read payload scan")?;
            if n == 0 {
                return Err(GgufInspectionError::SourceChanged { field: "EOF during payload scan" });
            }
            filled = filled.checked_add(n).ok_or_else(|| overflow("payload chunk fill"))?;
        }
        for scalar in scratch[..chunk].chunks_exact(width) {
            let value = match dtype {
                GgmlDType::F16 => half::f16::from_le_bytes([scalar[0], scalar[1]]).to_f32_const(),
                GgmlDType::BF16 => half::bf16::from_le_bytes([scalar[0], scalar[1]]).to_f32_const(),
                // Width was selected above; only F32 reaches this branch.
                _ => f32::from_le_bytes([scalar[0], scalar[1], scalar[2], scalar[3]]),
            };
            if !value.is_finite() {
                return Err(GgufInspectionError::NonFinitePayload { tensor: bounded(name), element });
            }
            if !head && !conversion_is_finite(value, requested) {
                return Err(GgufInspectionError::NonFiniteConversion { tensor: bounded(name), element, requested });
            }
            element = add(element, 1)?;
        }
        remaining = remaining.checked_sub(integer(chunk)?).ok_or_else(|| overflow("payload remaining"))?;
    }
    Ok(())
}

// Frozen software scalar round-to-nearest, ties-to-even policy. The actual
// rounded result is checked, including overflow caused by rounding. Finite
// underflow, signed zero and subnormals are accepted. Future assembly must use
// these same scalar conversions and retain their bits, never a bulk cast.
fn conversion_is_finite(value: f32, requested: Precision) -> bool {
    match requested {
        Precision::F16 => half::f16::from_f32_const(value).is_finite(),
        Precision::Bf16 => half::bf16::from_f32_const(value).is_finite(),
        Precision::F32 => true, // Preserve the source dtype; widening is later.
    }
}

struct SourceStamp {
    dev: u64,
    ino: u64,
    file_type: u32,
    len: u64,
    mtime: (i64, i64),
    ctime: (i64, i64),
}

impl SourceStamp {
    fn capture(file: &File, operation: &'static str) -> Result<Self, GgufInspectionError> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let m = file.metadata().map_err(|source| io_error(operation, source))?;
            Ok(Self {
                dev: m.dev(),
                ino: m.ino(),
                file_type: m.mode() & 0o170000,
                len: m.len(),
                mtime: (m.mtime(), m.mtime_nsec()),
                ctime: (m.ctime(), m.ctime_nsec()),
            })
        }
        #[cfg(not(unix))]
        {
            let _ = (file, operation);
            Err(GgufInspectionError::UnsupportedHost)
        }
    }

    fn compare(&self, after: &Self) -> Result<(), GgufInspectionError> {
        for (changed, field) in [
            (self.dev != after.dev, "device"),
            (self.ino != after.ino, "inode"),
            (self.file_type != after.file_type, "file type"),
            (self.len != after.len, "length"),
            (self.mtime != after.mtime, "mtime"),
            (self.ctime != after.ctime, "ctime"),
        ] {
            if changed {
                return Err(GgufInspectionError::SourceChanged { field });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Expected {
    U32(u32),
    U16(u16),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(&'static str),
}
impl Expected {
    fn type_name(self) -> &'static str {
        match self {
            Self::U32(_) => "U32",
            Self::U16(_) => "U16",
            Self::I32(_) => "I32",
            Self::F32(_) => "F32",
            Self::Bool(_) => "Bool",
            Self::String(_) => "String",
        }
    }
    fn same_type(self, v: &Value) -> bool {
        matches!(
            (self, v),
            (Self::U32(_), Value::U32(_))
                | (Self::U16(_), Value::U16(_))
                | (Self::I32(_), Value::I32(_))
                | (Self::F32(_), Value::F32(_))
                | (Self::Bool(_), Value::Bool(_))
                | (Self::String(_), Value::String(_))
        )
    }
    fn matches(self, v: &Value) -> bool {
        match (self, v) {
            (Self::U32(a), Value::U32(b)) => a == *b,
            (Self::U16(a), Value::U16(b)) => a == *b,
            (Self::I32(a), Value::I32(b)) => a == *b,
            (Self::F32(a), Value::F32(b)) => a == *b,
            (Self::Bool(a), Value::Bool(b)) => a == *b,
            (Self::String(a), Value::String(b)) => a == b,
            _ => false,
        }
    }
}
impl fmt::Display for Expected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::U32(v) => write!(f, "{v}"),
            Self::U16(v) => write!(f, "{v}"),
            Self::I32(v) => write!(f, "{v}"),
            Self::F32(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::String(v) => write!(f, "{}", Text(v)),
        }
    }
}

fn check_value(key: &str, value: &Value, expected: Expected) -> Result<(), GgufInspectionError> {
    if !expected.same_type(value) {
        return Err(GgufInspectionError::MetadataType {
            key: bounded(key),
            expected: expected.type_name(),
            actual: value.value_type(),
        });
    }
    if !expected.matches(value) {
        // Only scalar/short text diagnostics, never Value's recursive Debug.
        let actual = match value {
            Value::U32(v) => v.to_string(),
            Value::U16(v) => v.to_string(),
            Value::I32(v) => v.to_string(),
            Value::F32(v) => v.to_string(),
            Value::Bool(v) => v.to_string(),
            Value::String(v) => bounded(v),
            _ => String::new(), // same_type above excludes arrays and other variants.
        };
        return Err(GgufInspectionError::MetadataValue { key: bounded(key), expected: expected.to_string(), actual });
    }
    Ok(())
}

fn config_u32(field: &'static str, value: usize) -> Result<Expected, GgufInspectionError> {
    let value = u32::try_from(value).map_err(|_| GgufInspectionError::CompanionConfig { field })?;
    if value == 0 {
        return Err(GgufInspectionError::CompanionConfig { field });
    }
    Ok(Expected::U32(value))
}
fn config_f32(field: &'static str, value: f64) -> Result<Expected, GgufInspectionError> {
    let rounded = value as f32;
    if !value.is_finite() || value <= 0.0 || !rounded.is_finite() || rounded <= 0.0 {
        return Err(GgufInspectionError::CompanionConfig { field });
    }
    Ok(Expected::F32(rounded))
}

fn check_metadata(
    content: &Content,
    cfg: &LlamaConfig,
    outtype: GgufOuttype,
    count: u64,
) -> Result<(), GgufInspectionError> {
    // G1b2 owns default/full RoPE, causal, GQA, untied and uniform bias guards.
    // Borrow that validated configuration; never replace its f64 values.
    let required = [
        ("general.architecture", Expected::String("llama")),
        ("llama.block_count", config_u32("layers", cfg.num_layers)?),
        ("llama.embedding_length", config_u32("hidden", cfg.hidden)?),
        ("llama.feed_forward_length", config_u32("intermediate", cfg.intermediate)?),
        ("llama.vocab_size", config_u32("vocab", cfg.vocab)?),
        ("llama.context_length", config_u32("max_positions", cfg.max_positions)?),
        ("llama.attention.head_count", config_u32("heads", cfg.heads)?),
        ("llama.attention.head_count_kv", config_u32("kv_heads", cfg.kv_heads)?),
        ("llama.attention.key_length", config_u32("head_dim", cfg.head_dim)?),
        ("llama.attention.value_length", config_u32("head_dim", cfg.head_dim)?),
        ("llama.rope.dimension_count", config_u32("head_dim", cfg.head_dim)?),
        ("llama.attention.layer_norm_rms_epsilon", config_f32("rms_eps", cfg.rms_eps)?),
        ("llama.rope.freq_base", config_f32("rope_theta", cfg.rope_theta)?),
        (
            "general.file_type",
            Expected::U32(match outtype {
                GgufOuttype::F32 => 0,
                GgufOuttype::F16 => 1,
                GgufOuttype::Bf16 => 32,
            }),
        ),
    ];
    for &(key, expected) in &required {
        let value = content.metadata.get(key).ok_or(GgufInspectionError::MissingMetadata { key })?;
        check_value(key, value, expected)?;
    }
    let optional = [
        ("general.alignment", Expected::U32(32)),
        ("general.type", Expected::String("model")),
        ("llama.attention.causal", Expected::Bool(true)),
        ("llama.rope.scaling.type", Expected::String("none")),
        ("split.no", Expected::U16(0)),
        ("split.count", Expected::U16(1)),
        ("split.tensors.count", Expected::I32(i32::try_from(count).map_err(|_| overflow("split tensor count"))?)),
        ("general.quantization_version", Expected::U32(2)),
    ];
    for &(key, expected) in &optional {
        if let Some(value) = content.metadata.get(key) {
            check_value(key, value, expected)?;
        }
    }
    // Select a borrowed key, never a copied expected map or HashMap-dependent
    // diagnostic. All remaining passive validation is deterministic as well.
    let remaining = |key: &str| !required.iter().chain(&optional).any(|(known, _)| *known == key);
    if let Some(key) = content.metadata.keys().filter(|k| remaining(k) && passive_kind(k).is_none()).min() {
        return Err(if key.starts_with("split.") {
            GgufInspectionError::SplitUnsupported { key: bounded(key) }
        } else {
            GgufInspectionError::UnsupportedMetadata { key: bounded(key) }
        });
    }
    if let Some((key, value, kind)) = content
        .metadata
        .iter()
        .filter(|(k, _)| remaining(k))
        .filter_map(|(k, v)| passive_kind(k).filter(|kind| !kind.matches(v)).map(|kind| (k, v, kind)))
        .min_by(|(a, ..), (b, ..)| a.cmp(b))
    {
        // Report only the type; never copy untrusted values or arrays into the error.
        return Err(GgufInspectionError::MetadataType {
            key: bounded(key),
            expected: kind.name(),
            actual: value.value_type(),
        });
    }
    Ok(())
}

enum PassiveKind {
    String,
    Strings,
    U32,
    Tokenizer,
}
impl PassiveKind {
    fn matches(&self, value: &Value) -> bool {
        match self {
            Self::String => matches!(value, Value::String(_)),
            Self::Strings => matches!(value, Value::Array(v) if v.iter().all(|v| matches!(v, Value::String(_)))),
            Self::U32 => matches!(value, Value::U32(_)),
            Self::Tokenizer => true,
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::String => "String",
            Self::Strings => "Array of String",
            Self::U32 => "U32",
            Self::Tokenizer => "any bounded Value",
        }
    }
}
fn passive_kind(key: &str) -> Option<PassiveKind> {
    if key.starts_with("tokenizer.") {
        return Some(PassiveKind::Tokenizer);
    }
    match key {
        "general.name"
        | "general.author"
        | "general.version"
        | "general.organization"
        | "general.finetune"
        | "general.basename"
        | "general.description"
        | "general.quantized_by"
        | "general.size_label"
        | "general.license"
        | "general.url"
        | "general.doi"
        | "general.uuid"
        | "general.repo_url"
        | "general.license.name"
        | "general.license.link"
        | "general.source.url"
        | "general.source.doi"
        | "general.source.uuid"
        | "general.source.repo_url" => return Some(PassiveKind::String),
        "general.tags" | "general.languages" => return Some(PassiveKind::Strings),
        "general.base_model.count" | "general.dataset.count" => return Some(PassiveKind::U32),
        _ => (),
    }
    let rest = key.strip_prefix("general.base_model.").or_else(|| key.strip_prefix("general.dataset."))?;
    let (id, suffix) = rest.split_once('.')?;
    if id.is_empty()
        || !id.bytes().all(|b| b.is_ascii_digit())
        || (id.len() > 1 && id.starts_with('0'))
        || id.parse::<u32>().is_err()
    {
        return None;
    }
    match suffix {
        "name" | "author" | "version" | "organization" | "description" | "url" | "doi" | "uuid" | "repo_url" => {
            Some(PassiveKind::String)
        }
        _ => None,
    }
}

fn matrix_type(outtype: GgufOuttype) -> (GgmlDType, u64) {
    match outtype {
        GgufOuttype::F16 => (GgmlDType::F16, 2),
        GgufOuttype::Bf16 => (GgmlDType::BF16, 2),
        GgufOuttype::F32 => (GgmlDType::F32, 4),
    }
}

fn check_inventory(
    content: &Content,
    cfg: &LlamaConfig,
    outtype: GgufOuttype,
    count: u64,
    header_end: u64,
    eof: u64,
) -> Result<u64, GgufInspectionError> {
    let actual = integer(content.tensor_infos.len())?;
    if actual != count {
        return Err(GgufInspectionError::TensorCount { expected: count, actual });
    }
    let base = mul(add(header_end, 31)? / 32, 32)?;
    if content.tensor_data_offset != base || base > eof {
        return Err(GgufInspectionError::DataOffset { expected: base, actual: content.tensor_data_offset });
    }
    let scratch = add(mul(count, integer(std::mem::size_of::<(u64, u64)>())?)?, 64)?;
    limit("span scratch bytes", scratch, MAX_SPAN_SCRATCH)?;
    let capacity = usize::try_from(count).map_err(|_| overflow("span capacity"))?;
    let mut spans = Vec::new();
    spans.try_reserve_exact(capacity).map_err(|source| GgufInspectionError::Allocation { source })?;
    let mut name = String::new();
    name.try_reserve_exact(LOOKUP_BYTES).map_err(|source| GgufInspectionError::Allocation { source })?;
    let (_, width) = matrix_type(outtype);
    let (h, i, v, l) =
        (integer(cfg.hidden)?, integer(cfg.intermediate)?, integer(cfg.vocab)?, integer(cfg.num_layers)?);
    let k = mul(integer(cfg.kv_heads)?, integer(cfg.head_dim)?)?;
    let mut source_bytes = 0;
    // Fixed MAPPING order. Equal cardinality plus every required lookup proves
    // the exact set, including all seven biases iff the retained config has bias.
    visit_roles(cfg, outtype, &mut name, |name, shape, dtype, width, _head| {
        let info = content
            .tensor_infos
            .get(name)
            .ok_or_else(|| GgufInspectionError::MissingTensor { tensor: bounded(name) })?;
        if info.ggml_dtype != dtype {
            return Err(GgufInspectionError::TensorType {
                tensor: bounded(name),
                expected: dtype,
                actual: info.ggml_dtype,
            });
        }
        // Candle has already reversed wire dimensions once. Never reverse or
        // transpose here; compare the logical dimensions directly.
        let dims = info.shape.dims();
        if dims.len() != shape.len()
            || dims.iter().zip(shape).any(|(&a, &b)| b == 0 || u64::try_from(a).ok() != Some(b))
        {
            return Err(GgufInspectionError::TensorShape { tensor: bounded(name) });
        }
        let elements = shape.iter().try_fold(1, |n, &dim| mul(n, dim))?;
        let bytes = mul(elements, width)?;
        limit("source tensor bytes", bytes, MAX_TENSOR_BYTES)?;
        let start = add(base, info.offset)?;
        let end = add(start, bytes)?;
        if info.offset % 32 != 0 || start % 32 != 0 {
            return Err(GgufInspectionError::TensorAlignment { tensor: bounded(name) });
        }
        if start < base || start >= end || end > eof {
            return Err(GgufInspectionError::TensorSpan { tensor: bounded(name), start, end });
        }
        source_bytes = add(source_bytes, bytes)?;
        // Exactly count distinct fixed names are visited, so the prior fallible
        // reservation suffices for every push; no growth allocation is required.
        spans.push((start, end));
        Ok(())
    })?;
    spans.sort_unstable_by_key(|&(start, end)| (start, end));
    for pair in spans.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(GgufInspectionError::OverlappingTensors { previous_end: pair[0].1, next_start: pair[1].0 });
        }
    }
    // Crosscheck the actual per-role sum against the closed dense formula.
    let matrix_elements =
        add(mul(mul(2, v)?, h)?, mul(l, add(add(mul(mul(2, h)?, h)?, mul(mul(2, h)?, k)?)?, mul(mul(3, h)?, i)?)?)?)?;
    let bias_per_layer = add(add(mul(3, h)?, mul(2, k)?)?, mul(2, i)?)?;
    let vectors = add(mul(add(mul(2, l)?, 1)?, h)?, if cfg.bias { mul(l, bias_per_layer)? } else { 0 })?;
    let expected = add(mul(width, matrix_elements)?, mul(4, vectors)?)?;
    check_source_bytes(expected, source_bytes)?;
    Ok(source_bytes)
}

// Fixed MAPPING order shared by header and payload phases. The caller reserves
// the lookup string; there is no allocated role inventory. Only output.weight
// is the source-preserving head, independently of requested precision.
fn visit_roles(
    cfg: &LlamaConfig,
    outtype: GgufOuttype,
    name: &mut String,
    mut visit: impl FnMut(&str, &[u64], GgmlDType, u64, bool) -> Result<(), GgufInspectionError>,
) -> Result<(), GgufInspectionError> {
    let (matrix, width) = matrix_type(outtype);
    let (h, i, v) = (integer(cfg.hidden)?, integer(cfg.intermediate)?, integer(cfg.vocab)?);
    let k = mul(integer(cfg.kv_heads)?, integer(cfg.head_dim)?)?;
    visit("token_embd.weight", &[v, h], matrix, width, false)?;
    visit("output_norm.weight", &[h], GgmlDType::F32, 4, false)?;
    visit("output.weight", &[v, h], matrix, width, true)?;
    let weights: [(&str, &[u64], GgmlDType, u64); 9] = [
        ("attn_norm.weight", &[h], GgmlDType::F32, 4),
        ("attn_q.weight", &[h, h], matrix, width),
        ("attn_k.weight", &[k, h], matrix, width),
        ("attn_v.weight", &[k, h], matrix, width),
        ("attn_output.weight", &[h, h], matrix, width),
        ("ffn_norm.weight", &[h], GgmlDType::F32, 4),
        ("ffn_gate.weight", &[i, h], matrix, width),
        ("ffn_up.weight", &[i, h], matrix, width),
        ("ffn_down.weight", &[h, i], matrix, width),
    ];
    let biases: [(&str, &[u64], GgmlDType, u64); 7] = [
        ("attn_q.bias", &[h], GgmlDType::F32, 4),
        ("attn_k.bias", &[k], GgmlDType::F32, 4),
        ("attn_v.bias", &[k], GgmlDType::F32, 4),
        ("attn_output.bias", &[h], GgmlDType::F32, 4),
        ("ffn_gate.bias", &[i], GgmlDType::F32, 4),
        ("ffn_up.bias", &[i], GgmlDType::F32, 4),
        ("ffn_down.bias", &[h], GgmlDType::F32, 4),
    ];
    let biases: &[_] = if cfg.bias { &biases } else { &[] };
    for layer in 0..cfg.num_layers {
        for &(role, shape, dtype, role_width) in weights.iter().chain(biases) {
            name.clear();
            // String's fmt::Write cannot fail. Fixed suffixes plus <=8195's
            // layer index fit <128 bytes, already fallibly reserved by the caller.
            let _ = write!(name, "blk.{layer}.{role}");
            visit(name, shape, dtype, role_width, false)?;
        }
    }
    Ok(())
}
