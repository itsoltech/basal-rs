//! Partial, host-only inspection of one explicitly selected dense Llama GGUF.
//!
//! This checks a bounded header and inventory, never payloads or the declared
//! digest. It grants no materialization, inference or memory-admission authority.

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

const MAX_FILE_BYTES: u64 = 64 << 30;
const MAX_TENSOR_BYTES: u64 = 8 << 30;
const MAX_TENSORS: u64 = 8195;
const MAX_SPAN_SCRATCH: u64 = 1 << 20;
const LOOKUP_BYTES: usize = 128;

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
    // Intentionally retained as a single ownership boundary for subsequent
    // authorized phases; no reads or source claims are made by these fields.
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
}

/// Implemented host/header/inventory refusals only.
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
    let (matrix, width) = matrix_type(outtype);
    let (h, i, v, l) =
        (integer(cfg.hidden)?, integer(cfg.intermediate)?, integer(cfg.vocab)?, integer(cfg.num_layers)?);
    let k = mul(integer(cfg.kv_heads)?, integer(cfg.head_dim)?)?;
    let mut source_bytes = 0;
    // Fixed MAPPING order. Equal cardinality plus every required lookup proves
    // the exact set, including all seven biases iff the retained config has bias.
    let mut visit = |name: &str, shape: &[u64], dtype: GgmlDType, width: u64| -> Result<(), GgufInspectionError> {
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
    };
    visit("token_embd.weight", &[v, h], matrix, width)?;
    visit("output_norm.weight", &[h], GgmlDType::F32, 4)?;
    visit("output.weight", &[v, h], matrix, width)?;
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
            // layer index fit <128 bytes, already fallibly reserved above.
            let _ = write!(name, "blk.{layer}.{role}");
            visit(&name, shape, dtype, role_width)?;
        }
    }
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
    if source_bytes != expected {
        return Err(GgufInspectionError::SourceBytes { expected, actual: source_bytes });
    }
    Ok(source_bytes)
}
