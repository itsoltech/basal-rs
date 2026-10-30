//! Custom full Llama thinking checkpoint (`think_source`): a local directory whose consumed files are named, sized
//! and hashed by an operator's `basal-think-source.1` integrity descriptor (`think_integrity`). The digests prove only
//! that basal read the bytes the operator declared: not publisher authenticity, training lineage or quality.
//!
//! Every check here runs before any GEMM table, device query or upload. The weight files stay open in one
//! [`Inventory`] (header, hash and every tensor read through the same handle); every other file is read once,
//! hashed and closed, and only its stamp is kept for [`VerifiedSource::fence`]. The operator keeps the directory
//! unchanged while the process runs (stable-directory contract, docs/THINKING.md); stamps do not make it an
//! immutable snapshot.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};
use basal_core::architecture::Architecture;
use basal_core::calibration::CalibrationSource;
use basal_core::checkpoint::{is_plain_file_name, CheckpointIndex, Inventory, Stamp, UniqueMap, MAX_INDEX_BYTES};
use basal_core::manifest::{LlamaConfig, ModelLoadOptions};
use basal_core::prompt::{ChatStyle, PromptLayout};
use basal_core::ModelManifest;
use serde::de::{Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::gemm_artifact::{ResolvedGemm, TableRequest};
use crate::model_options::{content_hash, llama_decoder, LoadedThinking, ModelOptions, ThinkNumerics};

const SCHEMA: &str = "basal-think-source.1";
const MODEL_ID_SCHEMA: &str = "basal-think-model.1";
const MIB: u64 = 1 << 20;
const STRUCTURAL: [&str; 4] = ["config.json", "basal.json", "tokenizer_config.json", "chat_template.jinja"];
const TOKENIZER: &str = "tokenizer.json";
const GENERATION: &str = "generation_config.json";
const SINGLE: &str = "model.safetensors";
const INDEX: &str = "model.safetensors.index.json";

/// `think_integrity` (`basal-think-source.1`); unknown fields and repeated fields or file names are refused.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    schema: String,
    files: UniqueMap<FileEntry>,
    calibration: CalibrationEntry,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileEntry {
    sha256: String,
    bytes: u64,
}

/// `"off"` or `{"file", "source_model_id"}` (each once, strings); anything else is `Invalid`.
enum CalibrationEntry {
    Off,
    File { file: String, source_model_id: String },
    Invalid,
}

/// Any JSON value, keeping only an unsigned integer or a string (an `arbitrary_precision` float or out-of-range
/// integer arrives as a map and is `Other`).
enum Loose {
    Unsigned(u64),
    Text(String),
    Other,
}

/// `generation_config.json`: unknown keys ignored, repeated known keys refused.
#[derive(Deserialize)]
struct Generation {
    eos_token_id: EosIds,
    bos_token_id: Option<u32>,
}

/// `eos_token_id`: one id or a list of ids; None for any other value.
struct EosIds(Option<Vec<u32>>);

/// Visitor methods for the scalar values a [`Visitor`] of this module maps to one constant.
macro_rules! other_scalars {
    ($value:expr) => {
        fn visit_i64<E>(self, _: i64) -> Result<Self::Value, E> {
            Ok($value)
        }
        fn visit_f64<E>(self, _: f64) -> Result<Self::Value, E> {
            Ok($value)
        }
        fn visit_bool<E>(self, _: bool) -> Result<Self::Value, E> {
            Ok($value)
        }
        fn visit_unit<E>(self) -> Result<Self::Value, E> {
            Ok($value)
        }
    };
}

/// Consume every remaining element or entry without keeping it.
fn drain_seq<'de, A: SeqAccess<'de>>(mut seq: A) -> Result<(), A::Error> {
    while seq.next_element::<IgnoredAny>()?.is_some() {}
    Ok(())
}

fn drain_map<'de, A: MapAccess<'de>>(mut map: A) -> Result<(), A::Error> {
    while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
    Ok(())
}

impl<'de> Deserialize<'de> for Loose {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Loose;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_u64<E>(self, v: u64) -> Result<Loose, E> {
                Ok(Loose::Unsigned(v))
            }
            fn visit_str<E>(self, v: &str) -> Result<Loose, E> {
                Ok(Loose::Text(v.to_string()))
            }
            other_scalars!(Loose::Other);
            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Loose, A::Error> {
                drain_seq(seq)?;
                Ok(Loose::Other)
            }
            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Loose, A::Error> {
                drain_map(map)?;
                Ok(Loose::Other)
            }
        }
        deserializer.deserialize_any(V)
    }
}

impl<'de> Deserialize<'de> for CalibrationEntry {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = CalibrationEntry;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(r#""off" or {"file","source_model_id"}"#)
            }
            fn visit_str<E>(self, v: &str) -> Result<CalibrationEntry, E> {
                Ok(if v == "off" { CalibrationEntry::Off } else { CalibrationEntry::Invalid })
            }
            fn visit_u64<E>(self, _: u64) -> Result<CalibrationEntry, E> {
                Ok(CalibrationEntry::Invalid)
            }
            other_scalars!(CalibrationEntry::Invalid);
            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<CalibrationEntry, A::Error> {
                drain_seq(seq)?;
                Ok(CalibrationEntry::Invalid)
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<CalibrationEntry, A::Error> {
                let (mut file, mut source_model_id, mut valid) = (None, None, true);
                while let Some(key) = map.next_key::<String>()? {
                    let value = map.next_value::<Loose>()?;
                    let slot = match key.as_str() {
                        "file" => &mut file,
                        "source_model_id" => &mut source_model_id,
                        _ => {
                            valid = false;
                            continue;
                        }
                    };
                    match value {
                        Loose::Text(text) if slot.is_none() => *slot = Some(text),
                        _ => valid = false,
                    }
                }
                Ok(match (valid, file, source_model_id) {
                    (true, Some(file), Some(source_model_id)) => CalibrationEntry::File { file, source_model_id },
                    _ => CalibrationEntry::Invalid,
                })
            }
        }
        deserializer.deserialize_any(V)
    }
}

impl<'de> Deserialize<'de> for EosIds {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = EosIds;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("an integer or a list of integers")
            }
            fn visit_u64<E>(self, v: u64) -> Result<EosIds, E> {
                Ok(EosIds(u32::try_from(v).ok().map(|id| vec![id])))
            }
            fn visit_str<E>(self, _: &str) -> Result<EosIds, E> {
                Ok(EosIds(None))
            }
            other_scalars!(EosIds(None));
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<EosIds, A::Error> {
                let mut ids = Some(Vec::new());
                while let Some(item) = seq.next_element::<Loose>()? {
                    let id = match item {
                        Loose::Unsigned(v) => u32::try_from(v).ok(),
                        Loose::Text(_) | Loose::Other => None,
                    };
                    match (id, &mut ids) {
                        (Some(id), Some(list)) => list.push(id),
                        _ => ids = None,
                    }
                }
                Ok(EosIds(ids))
            }
            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<EosIds, A::Error> {
                drain_map(map)?;
                Ok(EosIds(None))
            }
        }
        deserializer.deserialize_any(V)
    }
}

/// `model_id`: SHA-256 of the compact JSON of this value over every consumed file except the calibration; names in
/// byte order, fields in declaration order, no local path.
#[derive(Serialize)]
struct ModelId<'a> {
    files: BTreeMap<&'a str, IdFile<'a>>,
    schema: &'static str,
}

#[derive(Serialize)]
struct IdFile<'a> {
    bytes: u64,
    sha256: &'a str,
}

/// Named source estimates of custom thinking memory, tied to the loader's allocation sites and computed with checked
/// arithmetic from the bounded configuration and the inventory. They exclude Vec growth beyond the stated factors,
/// header-parser amplification before admission, allocator and driver overhead, the base model, K/V and generation
/// work: not allocator, RSS or OOM bounds.
pub(crate) struct Resources {
    /// largest host transient while one tensor is assembled: 3S for a multi-part tensor (Vec growth while appending,
    /// then an f16 or realigned copy), 2S for a single part, lm_head 2S
    assembly: u64,
    /// host RoPE build: the f32 table, its `Arc` copy and the f64 inverse frequencies
    rope_transient: u64,
    /// RoPE table and inverse frequencies each loaded instance retains
    retained: u64,
    /// parsed inventory metadata (headers plus 128 bytes per tensor), released with the handles
    metadata: u64,
    /// largest raw safetensors header read at once; parsed entries can transiently use several times these bytes
    /// before name/rank/count validation, during Inventory::open and before admission
    header: u64,
    /// device bytes: one decoder layer, the embedding (first stage), one norm, lm_head (last stage) and the cuBLASLt
    /// workspace of a stage
    layer: u64,
    embedding: u64,
    norm: u64,
    lm_head: u64,
    workspace: u64,
}

fn overflow() -> anyhow::Error {
    anyhow::anyhow!("custom thinking resource arithmetic overflows")
}

fn product(factors: &[u64]) -> Result<u64> {
    factors.iter().try_fold(1u64, |p, &f| p.checked_mul(f)).ok_or_else(overflow)
}

fn sum(terms: &[u64]) -> Result<u64> {
    terms.iter().try_fold(0u64, |s, &t| s.checked_add(t)).ok_or_else(overflow)
}

impl Resources {
    fn new(cfg: &LlamaConfig) -> Result<Self> {
        let n = |v: usize| u64::try_from(v).map_err(|_| overflow());
        let (h, i, v, d, p) =
            (n(cfg.hidden)?, n(cfg.intermediate)?, n(cfg.vocab)?, n(cfg.head_dim)?, n(cfg.max_positions)?);
        let qd = product(&[n(cfg.heads)?, d])?;
        let kvd = product(&[n(cfg.kv_heads)?, d])?;
        let qkv = sum(&[qd, kvd, kvd])?;
        let s_qkv = product(&[2, qkv, h])?;
        let s_o = product(&[2, h, qd])?;
        let s_gu = product(&[4, i, h])?;
        let s_down = product(&[2, h, i])?;
        let s_emb = product(&[2, v, h])?;
        let rope = product(&[4, p, d])?;
        let inv_freq = product(&[8, d / 2])?;
        // every single host allocation: the growing parts buffer (< 2S), a converted copy (S), the RoPE table
        let allocation = [product(&[2, s_qkv])?, product(&[2, s_gu])?, s_o, s_down, s_emb, rope].into_iter().max();
        ensure!(
            allocation.is_some_and(|a| a <= isize::MAX as u64 && usize::try_from(a).is_ok()),
            "custom thinking resource arithmetic overflows"
        );
        let cuda = basal_gpu::plan::BackendKind::llama() == basal_gpu::plan::BackendKind::Cuda;
        let lt = cuda && basal_gpu::DiagnosticOverrides::process().gemm() != Some("cublas");
        Ok(Self {
            assembly: [product(&[3, s_qkv])?, product(&[3, s_gu])?, product(&[2, s_o])?, product(&[2, s_down])?]
                .into_iter()
                .chain([product(&[2, s_emb])?])
                .max()
                .unwrap_or(0),
            rope_transient: sum(&[rope, rope, inv_freq])?,
            retained: sum(&[rope, inv_freq])?,
            metadata: 0,
            header: 0,
            layer: product(&[
                2,
                sum(&[
                    product(&[4, h])?,
                    product(&[qkv, sum(&[h, 1])?])?,
                    product(&[h, qd])?,
                    product(&[2, i, sum(&[h, 1])?])?,
                    product(&[h, i])?,
                ])?,
            ])?,
            embedding: s_emb,
            norm: product(&[2, h])?,
            lm_head: s_emb,
            workspace: if lt { 32 * MIB } else { 0 },
        })
    }

    fn set_inventory(&mut self, inventory: &Inventory) -> Result<()> {
        let tensors = u64::try_from(inventory.tensor_count()).map_err(|_| overflow())?;
        self.metadata = sum(&[inventory.total_header_bytes(), product(&[128, tensors])?])?;
        self.header = inventory.max_header_bytes();
        Ok(())
    }

    /// Host bytes one instance needs while it loads: metadata plus the largest of assembly, RoPE build and header.
    pub(crate) fn host_instance(&self) -> Result<u64> {
        sum(&[self.metadata, self.assembly.max(self.rope_transient).max(self.header)])
    }

    /// Host bytes each loaded instance keeps (its RoPE table).
    pub(crate) fn retained_host(&self) -> u64 {
        self.retained
    }

    /// Device bytes of the stage holding `layers` (the first stage holds the embedding, every stage a norm: the next
    /// stage's input norm or the final norm, the last one lm_head) plus its cuBLASLt workspace. Every precision stores
    /// two bytes per weight; biases are always allocated. No evidence head.
    pub(crate) fn stage_bytes(&self, layers: Range<usize>, first: bool, last: bool) -> Result<u64> {
        let layers = u64::try_from(layers.len()).map_err(|_| overflow())?;
        sum(&[
            product(&[self.layer, layers])?,
            if first { self.embedding } else { 0 },
            self.norm,
            if last { self.lm_head } else { 0 },
            self.workspace,
        ])
    }
}

/// A verified custom checkpoint: its open weight inventory and the stamps of the closed non-weight files and the
/// descriptor. Dropping the last reference closes the weight handles.
pub(crate) struct VerifiedSource {
    pub(crate) inventory: Inventory,
    dir: PathBuf,
    descriptor: (PathBuf, Stamp),
    files: Vec<(String, Stamp)>,
    pub(crate) model_id: String,
    pub(crate) gemm_request: TableRequest,
    pub(crate) resources: Resources,
}

impl VerifiedSource {
    /// Stable-directory fence: every weight handle and path, every other consumed file and the descriptor still have
    /// the stamp of their first read. Detects replacement, truncation or a changed size/mtime/ctime, not a same-size
    /// write that preserves them, a replace-and-restore between checks or a change after the last fence.
    pub(crate) fn fence(&self) -> Result<()> {
        self.inventory.check_unchanged()?;
        for (name, stamp) in &self.files {
            stamp.check(&self.dir.join(name), name)?;
        }
        self.descriptor.1.check(&self.descriptor.0, "integrity descriptor")
    }
}

/// The custom source of a loaded thinking checkpoint and its own GEMM table (resolved after admission).
#[derive(Clone)]
pub(crate) struct CustomSource {
    pub(crate) verified: Arc<VerifiedSource>,
    pub(crate) gemm: Option<ResolvedGemm>,
}

impl CustomSource {
    /// Resolve `think_gemm_table` for the generic Llama profile and the custom checkpoint's own shapes on `devices`
    /// (never the base table) with its own projection shape policy `separate`; `auto` may search, so this runs only
    /// after custom validation and admission.
    pub(crate) fn resolve_gemm(
        &mut self,
        manifest: &ModelManifest,
        precision: basal_gpu::Precision,
        cache: &Path,
        devices: &[usize],
        separate: bool,
    ) -> Result<&ResolvedGemm> {
        let gemm = crate::gemm_artifact::resolve_for_projections(
            &self.verified.gemm_request,
            manifest,
            crate::models::generic_llama_profile(),
            precision,
            cache,
            devices,
            separate,
        )
        .with_context(|| format!("GEMM table of the custom thinking checkpoint {}", manifest.name))?;
        Ok(self.gemm.insert(gemm))
    }
}

/// The device and layer range of every stage of one instance, in layer order.
pub(crate) type Stages = Vec<(usize, Range<usize>)>;

/// Host and device admission of custom thinking instances, before any GEMM table or upload: `host` bytes (one
/// instance's [`Resources::host_instance`] or [`server_host`]) against the host, then on every device the stage
/// payloads of `instances` against its free memory; a unified device adds its payload to the host need. Source
/// estimates checked once, not allocator or OOM guarantees.
pub(crate) fn admit(host: u64, instances: &[(&Resources, Stages)]) -> Result<()> {
    check_host(host)?;
    let mut devices: BTreeMap<usize, (Vec<String>, u64)> = BTreeMap::new();
    for (resources, stages) in instances {
        let last = stages.len().saturating_sub(1);
        for (s, (device, layers)) in stages.iter().enumerate() {
            let bytes = resources.stage_bytes(layers.clone(), s == 0, s == last)?;
            let entry = devices.entry(*device).or_default();
            entry.0.push(s.to_string());
            entry.1 = sum(&[entry.1, bytes])?;
        }
    }
    let mut unified = 0u64;
    for (device, (stages, bytes)) in &devices {
        let gpu = basal_gpu::gpu_memory(*device).with_context(|| format!("custom thinking device {device}"))?;
        ensure!(
            *bytes <= gpu.available_bytes,
            "custom thinking stage {} needs {bytes} estimated bytes on device {device}; {} available (weights and \
             cuBLASLt workspace only; base model, KV, generation work, allocator and context overhead excluded)",
            stages.join(", "),
            gpu.available_bytes
        );
        if gpu.unified {
            unified = sum(&[unified, *bytes])?;
        }
    }
    if unified > 0 {
        check_host(sum(&[host, unified])?)?;
    }
    Ok(())
}

/// Host need of a server's custom instances: every instance's retained RoPE plus the largest load transient.
pub(crate) fn server_host(instances: &[&Resources]) -> Result<u64> {
    let mut retained = 0u64;
    let mut largest = 0u64;
    for resources in instances {
        retained = sum(&[retained, resources.retained_host()])?;
        largest = largest.max(resources.host_instance()?);
    }
    sum(&[retained, largest])
}

fn check_host(bytes: u64) -> Result<()> {
    let host = crate::benchmark::resources::host_memory()?;
    ensure!(
        !host.pressure && host.available_bytes >= bytes,
        "custom thinking needs {bytes} estimated host bytes to load; {} available (source estimate, not an \
         allocator/OOM bound)",
        host.available_bytes
    );
    Ok(())
}

fn is_sha256(hex: &str) -> bool {
    hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Read the declared file `name` of `dir` once: a regular file (symlinks followed) of its declared size within
/// `ceiling`, read to the end and hashed against the descriptor, then closed. Returns its bytes and stamp.
fn read_declared(dir: &Path, name: &str, entry: &FileEntry, ceiling: u64) -> Result<(Vec<u8>, Stamp)> {
    let mut file = std::fs::File::open(dir.join(name)).with_context(|| format!("think_source: opening {name}"))?;
    let meta = file.metadata().with_context(|| format!("think_source: fstat {name}"))?;
    ensure!(meta.is_file(), "{name} is not a regular file");
    let len = meta.len();
    ensure!(len <= ceiling, "{name}: {len} bytes, limit {ceiling}");
    ensure!(len == entry.bytes, "{name}: {len} bytes on disk, the descriptor declares {}", entry.bytes);
    let mut bytes = Vec::with_capacity(usize::try_from(len)?);
    (&mut file).take(len + 1).read_to_end(&mut bytes).with_context(|| format!("think_source: reading {name}"))?;
    let read = u64::try_from(bytes.len())?;
    ensure!(read == entry.bytes, "{name}: {read} bytes on disk, the descriptor declares {}", entry.bytes);
    let got = content_hash(&bytes);
    ensure!(got == entry.sha256, "{name}: content sha256 {got} differs from the descriptor");
    Ok((bytes, Stamp::of(&meta)))
}

/// The eleven `LlamaConfig` fields are equal (it has no `PartialEq`).
fn same_config(a: &LlamaConfig, b: &LlamaConfig) -> bool {
    a.num_layers == b.num_layers
        && a.hidden == b.hidden
        && a.intermediate == b.intermediate
        && a.heads == b.heads
        && a.kv_heads == b.kv_heads
        && a.head_dim == b.head_dim
        && a.vocab == b.vocab
        && a.rms_eps == b.rms_eps
        && a.rope_theta == b.rope_theta
        && a.max_positions == b.max_positions
        && a.bias == b.bias
}

/// Verify the custom thinking checkpoint of `dir` against the descriptor `descriptor_path` and build its thinking
/// state, before any GEMM table, device query or upload: descriptor and consumed file set, non-weight files, bounded
/// Llama configuration and resource terms, manifest and calibration lineage, prompt protocol, tokenizer, EOS, weight
/// inventory with shapes and digests, identity, then core thinking rules without a backend and the fence.
pub(crate) fn load_custom_thinking(
    options: &ModelOptions,
    base: &ModelManifest,
    dir: &Path,
    descriptor_path: &Path,
) -> Result<LoadedThinking> {
    // 1. The descriptor, its file names and roles.
    let (descriptor_bytes, descriptor_stamp) = {
        let mut file = std::fs::File::open(descriptor_path).context("think_integrity: opening the descriptor")?;
        let meta = file.metadata().context("think_integrity: fstat")?;
        ensure!(meta.is_file(), "think_integrity is not a regular file");
        ensure!(meta.len() <= MIB, "think_integrity exceeds 1 MiB");
        let mut bytes = Vec::new();
        (&mut file).take(MIB + 1).read_to_end(&mut bytes).context("think_integrity: reading")?;
        ensure!(u64::try_from(bytes.len())? <= MIB, "think_integrity exceeds 1 MiB");
        (bytes, Stamp::of(&meta))
    };
    let descriptor_sha256 = content_hash(&descriptor_bytes);
    let descriptor: Descriptor =
        serde_json::from_str(std::str::from_utf8(&descriptor_bytes).context("think_integrity is not UTF-8")?)
            .context("think_integrity")?;
    drop(descriptor_bytes);
    ensure!(descriptor.schema == SCHEMA, "think_integrity: unsupported schema {} ({SCHEMA})", descriptor.schema);
    let files = descriptor.files.into_map("think_integrity files")?;
    for (name, entry) in &files {
        ensure!(is_plain_file_name(name), "think_integrity: {name} is not a plain file name");
        ensure!(is_sha256(&entry.sha256), "think_integrity: sha256 of {name} must be 64 lowercase hex digits");
    }
    ensure!((7..=72).contains(&files.len()), "think_integrity: {} files, expected 7 to 72", files.len());
    let indexed = match (files.contains_key(SINGLE), files.contains_key(INDEX)) {
        (true, false) => false,
        (false, true) => true,
        _ => bail!("think_integrity: name exactly one of model.safetensors and model.safetensors.index.json"),
    };
    let root = if indexed { INDEX } else { SINGLE };
    let calibration = match descriptor.calibration {
        CalibrationEntry::Off => None,
        CalibrationEntry::File { file, source_model_id } => {
            ensure!(is_plain_file_name(&file), "think_integrity: {file} is not a plain file name");
            ensure!(is_sha256(&source_model_id), "think_integrity: source_model_id must be 64 lowercase hex digits");
            ensure!(
                !STRUCTURAL.contains(&file.as_str())
                    && ![TOKENIZER, GENERATION, SINGLE, INDEX].contains(&file.as_str()),
                "think_integrity: the calibration file {file} has another role"
            );
            Some((file, source_model_id))
        }
        CalibrationEntry::Invalid => bail!(r#"calibration must be "off" or {{"file","source_model_id"}}"#),
    };
    let entry = |name: &str| files.get(name).with_context(|| format!("think_integrity: missing {name}"));
    for name in STRUCTURAL.into_iter().chain([TOKENIZER, GENERATION, root]) {
        entry(name)?;
    }
    if let Some((file, _)) = &calibration {
        entry(file)?;
    }

    // 2. Every non-weight file: one read, hashed and closed; only the stamps stay.
    let mut stamps = Vec::new();
    let mut take = |name: &str, ceiling: u64| -> Result<Vec<u8>> {
        let (bytes, stamp) = read_declared(dir, name, entry(name)?, ceiling)?;
        stamps.push((name.to_string(), stamp));
        Ok(bytes)
    };
    let config_bytes = take("config.json", MIB)?;
    for name in &STRUCTURAL[1..] {
        take(name, MIB)?;
    }
    let tokenizer_bytes = take(TOKENIZER, 64 * MIB)?;
    let generation_bytes = take(GENERATION, MIB)?;
    let index = if indexed { Some(CheckpointIndex::parse(&take(INDEX, MAX_INDEX_BYTES)?)?) } else { None };
    if let Some((file, _)) = &calibration {
        take(file, 16 * MIB)?;
    }
    // The consumed set decides the weight files (the index in index mode, never directory contents) and must
    // equal the descriptor's names: no digest of a file that is not read.
    let shards: Vec<String> = match &index {
        Some(index) => index.shards().into_iter().map(str::to_string).collect(),
        None => vec![SINGLE.to_string()],
    };
    for shard in &shards {
        ensure!(
            !(indexed && shard == SINGLE),
            "think_integrity: name exactly one of model.safetensors and model.safetensors.index.json"
        );
        ensure!(
            calibration.as_ref().is_none_or(|(file, _)| file != shard),
            "think_integrity: the calibration file {shard} has another role"
        );
    }
    let consumed: BTreeSet<&str> = STRUCTURAL
        .into_iter()
        .chain([TOKENIZER, GENERATION, root])
        .chain(shards.iter().map(String::as_str))
        .chain(calibration.as_ref().map(|(file, _)| file.as_str()))
        .collect();
    for name in &consumed {
        entry(name)?;
    }
    for name in files.keys() {
        ensure!(consumed.contains(name.as_str()), "think_integrity: {name} is not consumed by the thinking checkpoint");
    }

    // 3. The bounded Llama configuration and every resource term.
    let config: Value = serde_json::from_slice(&config_bytes).context("config.json")?;
    let Architecture::Llama(cfg) = Architecture::parse(&config)? else {
        bail!("custom thinking requires a Llama checkpoint")
    };
    ensure!(cfg.head_dim == 128, "custom thinking requires head_dim 128 (fused attention kernels)");
    for (field, value, limit) in [
        ("num_attention_heads", cfg.heads, 256),
        ("hidden_size", cfg.hidden, 32768),
        ("num_hidden_layers", cfg.num_layers, 256),
        ("intermediate_size", cfg.intermediate, 262_144),
        ("vocab_size", cfg.vocab, 1 << 20),
        ("max_position_embeddings", cfg.max_positions, 1 << 20),
    ] {
        ensure!(
            value <= limit,
            "custom thinking dimensions exceed the supported bounds: {field} = {value}, limit {limit}"
        );
    }
    let mut resources = Resources::new(&cfg)?;

    // 4. The manifest (core re-reads its files by path) with canonical/order1 and the declared calibration only.
    let calibration_path = calibration.as_ref().map(|(file, _)| dir.join(file));
    let manifest = ModelManifest::load_with_options(
        dir,
        ModelLoadOptions {
            orders: Some("1"),
            layout: Some(PromptLayout::Canonical),
            calibration: calibration_path.as_deref().map_or(CalibrationSource::Off, CalibrationSource::File),
            strict_calibration: false,
            served_identity: None,
        },
    )
    .context("custom thinking manifest")?;
    let Architecture::Llama(loaded) = &manifest.config else { bail!("custom thinking requires a Llama checkpoint") };
    ensure!(same_config(&cfg, loaded), "config.json changed between reads (stable-directory contract)");
    ensure!(manifest.calibration_unvalidated.is_none(), "thinking calibration must fit canonical/order1");

    // 5. The calibration core parsed is the declared file's bytes.
    if let Some((file, _)) = &calibration {
        ensure!(
            manifest.calibration_provenance.content_sha256.as_deref() == Some(entry(file)?.sha256.as_str()),
            "thinking calibration content differs from the descriptor digest"
        );
    }

    // 6. The thinking prompt carries every base option.
    let (think_options, base_options) = (manifest.prompt_format.max_options(), base.prompt_format.max_options());
    ensure!(
        think_options >= base_options,
        "thinking prompt format allows {think_options} options, the base allows {base_options}"
    );

    // 7. Tokenizer: vocabulary, decoded-byte bound and BOS.
    let tokenizer = basal_core::tokenize::BasalTokenizer::load_for(&dir.join(TOKENIZER), ChatStyle::Llama)?;
    ensure!(tokenizer.vocab_size() == cfg.vocab, "thinking vocabulary size differs from config");
    let raw: Value = serde_json::from_slice(&tokenizer_bytes).context("tokenizer.json")?;
    drop(tokenizer_bytes);
    ensure!(
        raw.get("decoder") == Some(&llama_decoder()),
        "thinking decoded-byte bound requires the published Llama decoder"
    );
    drop(raw);
    let bos = tokenizer
        .token_id(&manifest.bos_token)
        .filter(|id| (*id as usize) < cfg.vocab)
        .with_context(|| format!("thinking BOS token {} is not a single vocabulary token", manifest.bos_token))?;
    drop(tokenizer);
    if let Some(id) = config.get("bos_token_id").filter(|v| !v.is_null()) {
        ensure!(
            id.as_u64() == Some(u64::from(bos)),
            "config.json bos_token_id {id} differs from the tokenizer BOS {bos}"
        );
    }

    // 8. EOS from the hashed generation_config.json bytes.
    let generation: Generation =
        serde_json::from_str(std::str::from_utf8(&generation_bytes).context("generation_config.json is not UTF-8")?)
            .context("generation_config.json")?;
    let Some(eos) = generation.eos_token_id.0 else {
        bail!("generation_config eos_token_id must be an integer or a list of integers")
    };
    ensure!(
        (1..=64).contains(&eos.len())
            && eos.iter().collect::<BTreeSet<_>>().len() == eos.len()
            && eos.iter().all(|id| (*id as usize) < cfg.vocab),
        "generation_config eos_token_id must list 1 to 64 unique ids below the vocabulary size"
    );
    if let Some(id) = generation.bos_token_id {
        ensure!(id == bos, "generation_config bos_token_id {id} differs from the tokenizer BOS {bos}");
    }

    // 9. Generation limits, capped by the checkpoint context.
    let settings = options.think_settings()?;
    basal_core::thinking::ThinkSettings::new(settings.tokens, settings.max_len.min(cfg.max_positions))?;

    // 10. The weight inventory: headers, routing, exact Llama tensors, then each file's digest by its handle.
    let max_tensors = cfg
        .num_layers
        .checked_mul(if cfg.bias { 16 } else { 9 })
        .and_then(|n| n.checked_add(3))
        .ok_or_else(overflow)?;
    let declared: Vec<(&str, u64)> = shards.iter().map(|s| Ok((s.as_str(), entry(s)?.bytes))).collect::<Result<_>>()?;
    let inventory = Inventory::open(dir, &declared, index.as_ref(), max_tensors)?;
    let expected = basal_gpu::llama_expected_tensors(&cfg);
    let missing: Vec<&String> = expected.keys().filter(|n| inventory.tensor(n).is_none()).collect();
    ensure!(missing.is_empty(), "custom thinking checkpoint lacks tensors {missing:?}");
    let extra: Vec<&str> = inventory.names().filter(|n| !expected.contains_key(*n)).collect();
    ensure!(extra.is_empty(), "checkpoint has tensors this runtime does not implement: {extra:?}");
    for (name, shape) in &expected {
        let got = &inventory.tensor(name).with_context(|| format!("checkpoint tensor {name} missing"))?.shape;
        ensure!(got == shape, "{name}: shape {got:?}, expected {shape:?}");
    }
    for shard in &shards {
        let got = inventory.sha256(shard)?;
        ensure!(got == entry(shard)?.sha256, "{shard}: content sha256 {got} differs from the descriptor");
    }
    resources.set_inventory(&inventory)?;

    // 11. Identity and calibration lineage.
    let calibration_file = calibration.as_ref().map(|(file, _)| file.as_str());
    let id = ModelId {
        files: files
            .iter()
            .filter(|(name, _)| Some(name.as_str()) != calibration_file)
            .map(|(name, e)| (name.as_str(), IdFile { bytes: e.bytes, sha256: &e.sha256 }))
            .collect(),
        schema: MODEL_ID_SCHEMA,
    };
    let model_id = content_hash(&serde_json::to_vec(&id)?);
    let (calibration_label, calibration_identity) = match &calibration {
        None => ("off".to_string(), json!({"off": true})),
        Some((file, source_model_id)) => {
            ensure!(
                *source_model_id == model_id,
                "thinking calibration source_model_id {source_model_id} differs from model_id {model_id}"
            );
            (file.clone(), json!({"file": file, "sha256": entry(file)?.sha256, "source_model_id": source_model_id}))
        }
    };
    let sources = json!({
        "kind": "operator-local",
        "provenance": "operator-supplied",
        "schema": SCHEMA,
        "descriptor_sha256": descriptor_sha256,
        "model_id": model_id,
        "profile": crate::models::generic_llama_profile().id,
        "files": files.iter().map(|(name, e)| (name.clone(), Value::from(e.sha256.clone()))).collect::<serde_json::Map<_, _>>(),
        "checkpoint": {
            "layout": if indexed { "index" } else { "single" },
            "shards": inventory.shard_count(),
            "tensors": inventory.tensor_count(),
            "bytes": inventory.total_bytes(),
        },
        "calibration": calibration_identity,
        "numerics": "inherit",
    });

    // 12. Every core thinking rule (canonical Llama order1, dense vocabulary, EOS, decoder bound), without a backend.
    drop(basal_core::thinking::Thinking::new(
        manifest.clone(),
        crate::NoBackend,
        settings,
        eos.clone(),
        calibration_label.clone(),
        sources.clone(),
    )?);

    // 13. Fence, then the verified source.
    let verified = VerifiedSource {
        inventory,
        dir: dir.to_path_buf(),
        descriptor: (descriptor_path.to_path_buf(), descriptor_stamp),
        files: stamps,
        model_id,
        gemm_request: options.think_gemm_request()?,
        resources,
    };
    verified.fence()?;
    crate::note!(
        "thinking: operator-supplied source verified (model_id {}, descriptor sha256 {descriptor_sha256}); digests \
         bind the bytes read, not publisher authenticity or quality",
        verified.model_id
    );
    Ok(LoadedThinking {
        manifest,
        settings,
        eos,
        calibration_label,
        sources,
        numerics: ThinkNumerics::Inherit,
        custom: Some(CustomSource { verified: Arc::new(verified), gemm: None }),
    })
}
