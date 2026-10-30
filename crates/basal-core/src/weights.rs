//! Typed GGUF provenance declarations and bounded companion byte snapshots.
//!
//! This module never opens GGUF weights or admits a model for execution. Hashes
//! bind captured bytes, not the truth of an operator's conversion attestation.

use std::collections::TryReserveError;
use std::fmt;
use std::fs::{File, Metadata};
use std::io::{self, Read};
use std::marker::PhantomData;
use std::path::Path;
use std::str::FromStr;

use serde::de::{value::MapAccessDeserializer, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};

mod calibration_inventory;
mod declaration_construction;
mod immutable_semantics;
mod json_envelope;
mod provider_construction;
mod raw_source;
mod semantic_json;
mod semantics;
pub use calibration_inventory::{
    CallerCalibrationDirectoryLease, GgufCalibrationInventory, GgufCalibrationInventoryError, GgufCalibrationVariant,
    GgufInventoryReadError, InventoryNameReason, InventoryOperation, InventoryPhase, InventoryReadMismatch,
    InventoryResource, InventoryStampField,
};
use declaration_construction::{DeclarationCharge, SourceBudget};
pub use declaration_construction::{GgufDeclarationConstructionError, GgufDeclarationSourceBudget};
pub use immutable_semantics::{
    GgufImmutableCalibrationProvenance, GgufImmutableCompanionSemantics, GgufImmutableIdentityBinding,
    GgufImmutableManifest, GgufImmutableSemanticError, ImmutableJsonRole,
};
use json_envelope::JsonReadMode;
pub use json_envelope::{JsonEnvelopeError, JsonEnvelopeResource};
pub use provider_construction::{
    GgufMergeScratchError, GgufMergeScratchReadError, GgufMergeTableError, GgufMergeTableReadError,
};
use raw_source::RawCharge;
pub use raw_source::{GgufRawSourceBudget, GgufRawSourceError};
pub use semantic_json::{
    GgufSemanticArray, GgufSemanticElements, GgufSemanticKind, GgufSemanticMembers, GgufSemanticObject,
    GgufSemanticValue, SemanticConstructionError,
};
pub use semantics::{
    CalibrationCapture, CalibrationDecision, CalibrationIdentityState, CalibrationMode, CalibrationProbe,
    CalibrationProbeRole, CalibrationReadRole, CapturedCalibrationFile, GgufCalibrationBinding,
    GgufCalibrationBindingOutcome, GgufCompanionSemantics, GgufIdentityBindingError, GgufSemanticError, LayoutCheck,
    OrdersCheck, OrdersVariantDecision, SemanticStage,
};

const DECLARATION: &str = "gguf-source.json";
const MAX_DECLARATION_BYTES: u64 = 64 * 1024;
const MAX_COMPANION_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TOTAL_COMPANION_BYTES: u64 = 128 * 1024 * 1024;
const CONVERTER_COMMIT: &str = "7df8b876b91208a86a6d792e1140b3deec14535b";
const EXPORTER_COMMIT: &str = "b20f0f9b159f71726459aa98ede1356ab65695fd";

/// A SHA-256 digest stored as 32 bytes; text input must be exactly 64 ASCII hex characters.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    /// Borrow the digest bytes without allocating or formatting them.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    fn of(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }
}

/// Invalid digest syntax. The offending input is deliberately not retained.
#[derive(Debug, thiserror::Error)]
#[error("SHA-256 must contain exactly 64 ASCII hexadecimal characters")]
pub struct InvalidSha256;

impl FromStr for Sha256Digest {
    type Err = InvalidSha256;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if text.len() != 64 || !text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(InvalidSha256);
        }
        let mut bytes = [0; 32];
        for (dest, pair) in bytes.iter_mut().zip(text.as_bytes().chunks_exact(2)) {
            let nibble = |byte: u8| match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                // The ASCII hex check above leaves only A..F here.
                _ => byte - b'A' + 10,
            };
            *dest = (nibble(pair[0]) << 4) | nibble(pair[1]);
        }
        Ok(Self(bytes))
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Sha256Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

/// The five fixed companion roles; these are not arbitrary filesystem references.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompanionRole {
    Config,
    Basal,
    Tokenizer,
    TokenizerConfig,
    ChatTemplate,
}

impl CompanionRole {
    const ALL: [Self; 5] = [Self::Config, Self::Basal, Self::Tokenizer, Self::TokenizerConfig, Self::ChatTemplate];

    /// The fixed filename read for this role.
    pub fn filename(self) -> &'static str {
        match self {
            Self::Config => "config.json",
            Self::Basal => "basal.json",
            Self::Tokenizer => "tokenizer.json",
            Self::TokenizerConfig => "tokenizer_config.json",
            Self::ChatTemplate => "chat_template.jinja",
        }
    }
}

/// The single supported, explicitly attested converter layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufLayout {
    LlamaCppHfLlamaV1,
}

/// Declared dense output type of the pinned ordinary converter branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufOuttype {
    F16,
    Bf16,
    F32,
}

/// Pinned tokenizer-only repair, including the selected GGUF copy/writer code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetadataRepair {
    TokenizerOnly,
}

/// A checked schema-1 attestation. Its declared GGUF digest is **unverified**.
///
/// Only a successful companion snapshot constructs this value. Private fields
/// cannot be mutated or deserialized independently to bypass validation.
pub struct GgufDeclaration {
    gguf_sha256: Sha256Digest,
    companion_sha256: [Sha256Digest; 5],
    outtype: GgufOuttype,
    model_id: String,
    revision: String,
    safetensors: Vec<OriginSafetensor>,
    metadata_repair: Option<MetadataRepair>,
    // Last: semantic backing (including every name) dies before its charge.
    _declaration_charge: Option<DeclarationCharge>,
}

struct OriginSafetensor {
    name: String,
    sha256: Sha256Digest,
}

impl GgufDeclaration {
    /// Explicit layout; never inferred from a filename or GGUF metadata.
    pub fn layout(&self) -> GgufLayout {
        GgufLayout::LlamaCppHfLlamaV1
    }

    /// Output type attested for the pinned ordinary converter branch.
    pub fn outtype(&self) -> GgufOuttype {
        self.outtype
    }

    /// Operator-declared GGUF digest; no GGUF bytes have been opened or checked.
    pub fn declared_gguf_sha256(&self) -> Sha256Digest {
        self.gguf_sha256
    }

    /// Declared digest, checked against captured bytes by the snapshot operation.
    pub fn companion_sha256(&self, role: CompanionRole) -> Sha256Digest {
        self.companion_sha256[role as usize]
    }

    /// Logical operator attestation; never resolved, downloaded or opened.
    pub fn origin_model_id(&self) -> &str {
        &self.model_id
    }

    /// Logical revision attestation, not a verified checkpoint revision.
    pub fn origin_revision(&self) -> &str {
        &self.revision
    }

    /// Unique logical filenames and declared digests in their original attested order.
    /// These digests have not been compared with source checkpoint files.
    pub fn origin_safetensors(&self) -> impl ExactSizeIterator<Item = (&str, Sha256Digest)> {
        self.safetensors.iter().map(|entry| (entry.name.as_str(), entry.sha256))
    }

    /// Whether the declaration attests the pinned tokenizer-only repair.
    pub fn metadata_repair(&self) -> Option<MetadataRepair> {
        self.metadata_repair
    }

    fn parse(bytes: &[u8]) -> Result<Self, WeightSourceError> {
        let Object(data): Object<DeclarationDto> = serde_json::from_slice(bytes).map_err(|source| {
            WeightSourceError::DeclarationJson { line: source.line(), column: source.column(), source }
        })?;
        exact(&data.schema, "basal.gguf-source.1", "schema")?;
        exact(&data.layout, "llama-cpp-hf-llama-v1", "layout")?;
        let gguf_sha256 = digest(&data.gguf_sha256, "gguf_sha256")?;
        let hashes = &data.companion_sha256.0;
        let companion_sha256 = [
            digest(&hashes.config, "companion_sha256.config.json")?,
            digest(&hashes.basal, "companion_sha256.basal.json")?,
            digest(&hashes.tokenizer, "companion_sha256.tokenizer.json")?,
            digest(&hashes.tokenizer_config, "companion_sha256.tokenizer_config.json")?,
            digest(&hashes.chat_template, "companion_sha256.chat_template.jinja")?,
        ];
        let converter = &data.converter.0;
        exact(&converter.repository, "ggml-org/llama.cpp", "converter.repository")?;
        exact(&converter.commit, CONVERTER_COMMIT, "converter.commit")?;
        exact(&converter.class, "conversion.llama.LlamaModel", "converter.class")?;
        exact(&converter.architecture, "LlamaForCausalLM", "converter.architecture")?;
        exact(&converter.endianness, "little", "converter.endianness")?;
        for (field, enabled) in [
            ("converter.split", converter.split),
            ("converter.tensor_overrides", converter.tensor_overrides),
            ("converter.weight_postprocessing", converter.weight_postprocessing),
        ] {
            if enabled {
                return invalid(field, "must be false");
            }
        }
        let outtype = match converter.outtype.as_str() {
            "f16" => GgufOuttype::F16,
            "bf16" => GgufOuttype::Bf16,
            "f32" => GgufOuttype::F32,
            _ => return invalid("converter.outtype", "expected f16, bf16 or f32"),
        };
        let origin = &data.origin.0;
        logical_origin(&origin.model_id, "origin.model_id")?;
        logical_origin(&origin.revision, "origin.revision")?;
        if origin.safetensors.is_empty() {
            return invalid("origin.safetensors", "must be nonempty");
        }
        // The entire declaration is bounded to 64 KiB. Compare borrowed names
        // without copying or reordering the attested inventory.
        for (index, entry) in origin.safetensors.iter().enumerate() {
            let name = &entry.0.name;
            if !name.ends_with(".safetensors")
                || name.contains(['/', '\\', ':'])
                || name.chars().any(char::is_control)
                || name == "."
                || name == ".."
            {
                return invalid("origin.safetensors.name", "expected one logical .safetensors filename");
            }
            if origin.safetensors[..index].iter().any(|previous| previous.0.name == *name) {
                return invalid("origin.safetensors.name", "duplicate filename");
            }
        }
        if let Some(Object(repair)) = &data.metadata_repair {
            exact(&repair.kind, "tokenizer-only", "metadata_repair.kind")?;
            exact(&repair.basal_exporter_commit, EXPORTER_COMMIT, "metadata_repair.basal_exporter_commit")?;
            exact(&repair.gguf_copy_commit, CONVERTER_COMMIT, "metadata_repair.gguf_copy_commit")?;
        }
        let origin = data.origin.0;
        let mut safetensors = Vec::new();
        safetensors.try_reserve_exact(origin.safetensors.len()).map_err(|source| WeightSourceError::Allocation {
            role: DECLARATION,
            bytes: origin.safetensors.len().saturating_mul(std::mem::size_of::<OriginSafetensor>()),
            source,
        })?;
        for Object(entry) in origin.safetensors {
            safetensors.push(OriginSafetensor {
                name: entry.name,
                sha256: digest(&entry.sha256, "origin.safetensors.sha256")?,
            });
        }
        Ok(Self {
            gguf_sha256,
            companion_sha256,
            outtype,
            model_id: origin.model_id,
            revision: origin.revision,
            safetensors,
            metadata_repair: data.metadata_repair.map(|_| MetadataRepair::TokenizerOnly),
            _declaration_charge: None,
        })
    }
}

/// Exact retained bytes of the declaration and five hash-verified companions.
///
/// No local paths are retained. Getters borrow these bytes without reopening
/// files. This is not a validated GGUF, model manifest, tokenizer or source plan.
pub struct GgufCompanionSnapshot {
    declaration: GgufDeclaration,
    declaration_bytes: Vec<u8>,
    declaration_sha256: Sha256Digest,
    companions: [Vec<u8>; 5],
    // Fields drop in declaration order: release only after all raw backing dies.
    _raw_charge: Option<RawCharge>,
}

impl GgufCompanionSnapshot {
    /// Read only six fixed filenames, opening each once and checking regular-file
    /// metadata before and after its bounded read. Does not enumerate `directory`.
    ///
    /// Limits: declaration 64 KiB, each companion 64 MiB, companions together
    /// 128 MiB. Allocation and I/O failures return typed errors. Byte limits are
    /// not physical OOM protection or limits on subsequent parser amplification.
    ///
    /// Captured bytes remain stable if paths change later. Open handles and
    /// metadata comparisons do not prevent malicious in-place writes, restored
    /// races or pathname traversal. The caller must trust the local file source.
    /// Future parsers must consume these bytes, not reopen the original paths
    /// while claiming the captured hashes still apply.
    pub fn read(directory: &Path) -> Result<Self, WeightSourceError> {
        Self::read_mode(directory, JsonReadMode::Legacy, SourceBudget::None)
    }

    fn read_mode(directory: &Path, mode: JsonReadMode, budget: SourceBudget<'_>) -> Result<Self, WeightSourceError> {
        // Locals drop in reverse order, so the charge must precede raw locals.
        let mut raw_charge = budget.raw().map(GgufRawSourceBudget::charge);
        let mut declaration_total = 0;
        let declaration_bytes = read_file(
            directory,
            DECLARATION,
            MAX_DECLARATION_BYTES,
            &mut declaration_total,
            MAX_DECLARATION_BYTES,
            raw_charge.as_mut(),
        )?;
        let declaration_sha256 = Sha256Digest::of(&declaration_bytes);
        mode.check(&declaration_bytes)
            .map_err(|source| WeightSourceError::JsonEnvelope { role: DECLARATION, source })?;
        let declaration = budget.declaration(&declaration_bytes)?;
        let mut companions = std::array::from_fn(|_| Vec::new());
        let mut total = 0;
        for role in CompanionRole::ALL {
            let bytes = read_file(
                directory,
                role.filename(),
                MAX_COMPANION_BYTES,
                &mut total,
                MAX_TOTAL_COMPANION_BYTES,
                raw_charge.as_mut(),
            )?;
            let actual = Sha256Digest::of(&bytes);
            let expected = declaration.companion_sha256(role);
            if actual != expected {
                return Err(WeightSourceError::CompanionDigest { role, expected, actual });
            }
            if role != CompanionRole::ChatTemplate {
                mode.check(&bytes)
                    .map_err(|source| WeightSourceError::JsonEnvelope { role: role.filename(), source })?;
            }
            companions[role as usize] = bytes;
        }
        Ok(Self { declaration, declaration_bytes, declaration_sha256, companions, _raw_charge: raw_charge })
    }

    /// Borrow the checked declaration; this does not validate the GGUF itself.
    pub fn declaration(&self) -> &GgufDeclaration {
        &self.declaration
    }

    /// Borrow the complete original declaration, without reserialization.
    pub fn declaration_bytes(&self) -> &[u8] {
        &self.declaration_bytes
    }

    /// Hash of the complete raw declaration, including whitespace and hex case.
    pub fn declaration_sha256(&self) -> Sha256Digest {
        self.declaration_sha256
    }

    /// Borrow captured, hash-verified bytes for a fixed role without I/O.
    pub fn bytes(&self, role: CompanionRole) -> &[u8] {
        &self.companions[role as usize]
    }
}

/// Failures of declaration validation or companion capture.
///
/// Display/Debug omit raw buffers and origin text; digest mismatches include
/// expected and captured hashes.
/// A JSON error's original parser cause remains accessible through `source()`;
/// callers must treat that detailed diagnostic as untrusted declaration data.
#[derive(thiserror::Error)]
pub enum WeightSourceError {
    #[error("gguf-source.json declaration construction failed")]
    DeclarationConstruction {
        #[source]
        source: GgufDeclarationConstructionError,
    },
    #[error("{role} raw source admission failed")]
    RawSource {
        role: &'static str,
        #[source]
        source: GgufRawSourceError,
    },
    #[error("{role} JSON envelope check failed")]
    JsonEnvelope {
        role: &'static str,
        #[source]
        source: JsonEnvelopeError,
    },
    #[error("invalid gguf-source.json JSON/schema at line {line}, column {column}")]
    DeclarationJson {
        line: usize,
        column: usize,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid gguf-source.json field {field}: {reason}")]
    InvalidDeclaration { field: &'static str, reason: &'static str },
    #[error("unsupported gguf-source.json field {field}: expected {expected}")]
    UnsupportedValue { field: &'static str, expected: &'static str },
    #[error("invalid gguf-source.json digest field {field}: {source}")]
    InvalidDigest {
        field: &'static str,
        #[source]
        source: InvalidSha256,
    },
    #[error("{role} is not a regular file")]
    NonRegular { role: &'static str },
    #[error("{role} changed while capturing bytes: {reason}")]
    SourceChanged { role: &'static str, reason: &'static str },
    #[error("{role} resource limit {limit} bytes: {reason}")]
    ResourceLimit { role: &'static str, limit: u64, reason: &'static str },
    #[error("{role}: reserving {bytes} bytes failed")]
    Allocation {
        role: &'static str,
        bytes: usize,
        #[source]
        source: TryReserveError,
    },
    #[error("{role}: {operation} failed: {source}")]
    Io {
        role: &'static str,
        operation: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("{role:?} companion digest mismatch: expected {expected}, captured {actual}")]
    CompanionDigest { role: CompanionRole, expected: Sha256Digest, actual: Sha256Digest },
}

impl fmt::Debug for WeightSourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

fn invalid<T>(field: &'static str, reason: &'static str) -> Result<T, WeightSourceError> {
    Err(WeightSourceError::InvalidDeclaration { field, reason })
}

fn digest(text: &str, field: &'static str) -> Result<Sha256Digest, WeightSourceError> {
    text.parse().map_err(|source| WeightSourceError::InvalidDigest { field, source })
}

fn exact(actual: &str, expected: &'static str, field: &'static str) -> Result<(), WeightSourceError> {
    if actual != expected {
        return Err(WeightSourceError::UnsupportedValue { field, expected });
    }
    Ok(())
}

fn logical_origin(value: &str, field: &'static str) -> Result<(), WeightSourceError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || value.chars().any(char::is_control)
        || trimmed.starts_with(['/', '\\'])
        || value.contains(':')
    {
        return invalid(field, "expected a nonempty logical origin, without controls, absolute paths or URI");
    }
    Ok(())
}

fn read_file(
    directory: &Path,
    role: &'static str,
    per_file_limit: u64,
    total: &mut u64,
    total_limit: u64,
    raw_charge: Option<&mut RawCharge>,
) -> Result<Vec<u8>, WeightSourceError> {
    read_path(&directory.join(role), role, per_file_limit, total, total_limit, raw_charge)
}

fn read_path(
    path: &Path,
    role: &'static str,
    per_file_limit: u64,
    total: &mut u64,
    total_limit: u64,
    raw_charge: Option<&mut RawCharge>,
) -> Result<Vec<u8>, WeightSourceError> {
    let io_error = |operation, source| WeightSourceError::Io { role, operation, source };
    let mut file = File::open(path).map_err(|source| io_error("open", source))?;
    let before = file.metadata().map_err(|source| io_error("metadata before read", source))?;
    if !before.is_file() {
        return Err(WeightSourceError::NonRegular { role });
    }
    if before.len() > per_file_limit {
        return Err(WeightSourceError::ResourceLimit {
            role,
            limit: per_file_limit,
            reason: "file length exceeds allowance",
        });
    }
    *total =
        total.checked_add(before.len()).filter(|sum| *sum <= total_limit).ok_or(WeightSourceError::ResourceLimit {
            role,
            limit: total_limit,
            reason: "cumulative length exceeds allowance or overflows",
        })?;
    let length = usize::try_from(before.len()).map_err(|_| WeightSourceError::ResourceLimit {
        role,
        limit: per_file_limit,
        reason: "file length cannot fit in usize",
    })?;
    let mut bytes = match raw_charge {
        Some(charge) => charge.allocate(length).map_err(|source| WeightSourceError::RawSource { role, source })?,
        None => {
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(length).map_err(|source| WeightSourceError::Allocation {
                role,
                bytes: length,
                source,
            })?;
            bytes
        }
    };
    let mut chunk = [0; 8192];
    while bytes.len() < length {
        let allowed = (length - bytes.len()).min(chunk.len());
        let count = file.read(&mut chunk[..allowed]).map_err(|source| io_error("read", source))?;
        if count == 0 {
            return Err(WeightSourceError::SourceChanged { role, reason: "EOF before initial length" });
        }
        // count <= allowed <= length-len: neither allocation route can grow.
        bytes.extend_from_slice(&chunk[..count]);
    }
    let mut probe = [0; 1];
    if file.read(&mut probe).map_err(|source| io_error("growth probe", source))? != 0 {
        return Err(WeightSourceError::SourceChanged { role, reason: "bytes beyond initial length" });
    }
    let after = file.metadata().map_err(|source| io_error("metadata after read", source))?;
    check_metadata(role, &before, &after)?;
    Ok(bytes)
}

fn check_metadata(role: &'static str, before: &Metadata, after: &Metadata) -> Result<(), WeightSourceError> {
    let modified = |metadata: &Metadata| {
        metadata.modified().map_err(|source| WeightSourceError::Io { role, operation: "modification time", source })
    };
    if !after.is_file() || before.len() != after.len() || modified(before)? != modified(after)? {
        return Err(WeightSourceError::SourceChanged {
            role,
            reason: "regular-file, length or modification-time metadata differs",
        });
    }
    Ok(())
}

// serde's derived struct visitor also accepts positional arrays. Every schema
// object uses this small map-only adapter so arrays cannot bypass JSON shape.
struct Object<T>(T);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Object<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ObjectVisitor<T>(PhantomData<T>);

        impl<'de, T: Deserialize<'de>> Visitor<'de> for ObjectVisitor<T> {
            type Value = Object<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a schema object")
            }

            fn visit_map<M: MapAccess<'de>>(self, map: M) -> Result<Self::Value, M::Error> {
                T::deserialize(MapAccessDeserializer::new(map)).map(Object)
            }
        }

        deserializer.deserialize_map(ObjectVisitor(PhantomData))
    }
}

// All untrusted declaration data remains private and is exposed only after
// validation. No DTO derives Debug or Serialize (in particular, origin values).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeclarationDto {
    schema: String,
    layout: String,
    gguf_sha256: String,
    converter: Object<ConverterDto>,
    companion_sha256: Object<CompanionHashesDto>,
    origin: Object<OriginDto>,
    // With deserialize_with and no default, serde rejects a missing field even
    // though the value itself can be explicit null. No Option missing fallback.
    #[serde(deserialize_with = "required_repair")]
    metadata_repair: Option<Object<RepairDto>>,
}

fn required_repair<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Object<RepairDto>>, D::Error> {
    Option::deserialize(deserializer)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConverterDto {
    repository: String,
    commit: String,
    class: String,
    architecture: String,
    outtype: String,
    endianness: String,
    split: bool,
    tensor_overrides: bool,
    weight_postprocessing: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompanionHashesDto {
    #[serde(rename = "config.json")]
    config: String,
    #[serde(rename = "basal.json")]
    basal: String,
    #[serde(rename = "tokenizer.json")]
    tokenizer: String,
    #[serde(rename = "tokenizer_config.json")]
    tokenizer_config: String,
    #[serde(rename = "chat_template.jinja")]
    chat_template: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginDto {
    model_id: String,
    revision: String,
    safetensors: Vec<Object<SafetensorsDto>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SafetensorsDto {
    name: String,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RepairDto {
    kind: String,
    basal_exporter_commit: String,
    gguf_copy_commit: String,
}
