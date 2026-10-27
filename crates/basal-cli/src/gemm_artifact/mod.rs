//! GEMM artifacts: acquisition and validation of the cuBLASLt table of a model instance, the first phase of its
//! execution plan (`basal_gpu::ResolvedExecutionPlan`).
//!
//! A table is identified by the SHA-256 of its normalized runtime content ([`normalize`]). Tables live in an
//! immutable content-addressed store `<gemm_cache>/artifacts/<sha256>.json`: a file is written once under a
//! unique temporary name and linked into place, never replaced; an existing file of another content is an error.
//! The bytes are stored as received (bundled reduced JSON, a search result, an imported file), so the file digest a
//! batch profile records equals that of the file used before. That SHA-256 of the bytes is captured with the snapshot
//! ([`GemmArtifact::bytes_sha256`]: of the store file, which an earlier artifact of the same normalized content keeps,
//! or of the bytes read); the batch profile never reads a table path again.
//!
//! A cache this process may not write (permission denied, read-only file system) does not stop a table already read
//! and validated (explicit, bundled, an earlier cache): it is used from memory, not stored, with a warning, and an
//! earlier cache is then not bound. Every other failure stays an error, among them an unreadable store or a stored
//! artifact of another content. A search and `gemm-digest --import` still need a writable store, and a `sha256:`
//! artifact that is not bundled must already be stored.
//!
//! `gemm_table` of a model:
//! - `auto` (default, adaptive): this machine's binding for the profile, model, dtype, GPU, cuBLASLt and search
//!   version (`<gemm_cache>/bindings/`), else, for a v1 compatibility profile only, the table of an earlier cache
//!   `<gemm_cache>/<model>-<dtype>.json` (imported into the store with its algorithms, the source left as it is),
//!   else the first valid bundled table the model profile pins, else Llama searches a table (stored and bound),
//!   Gemma uses fixed per-prompt GEMMs. Tables made or imported here are `adaptive/local` (`bundled` when the digest
//!   is a table of this build).
//! - `pinned` (reproducible): only a bundled table pinned by the model profile; none is an error.
//! - `sha256:<hex>` (reproducible): exactly that artifact, bundled or in the store; absent is an error.
//! - `none`: no table (Llama times heuristic algorithms at first use; not allowed with a reproducible plan).
//! - a path: that file (`explicit`, adaptive), read once and copied into the store when it is writable.
//!
//! Every table is read once; the parsed snapshot is what is validated, named by digest in the plan and loaded by
//! the backend. Before any weight is read it is checked on every GPU it will run on: cuBLASLt version and GPU name,
//! and cuBLASLt's acceptance of every algorithm when the plan is reproducible or the table is shared by several
//! GPUs. A reproducible table must also be batch-invariant, of this build's search version (a later one is not
//! understood by this build) and cover every weight shape of the model in the plan's dtype with entries of that
//! dtype only; at run time a Llama GEMM without a pinned algorithm is then an error, never timed.

pub mod normalize;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};
use basal_core::ModelManifest;
use basal_gpu::{GemmArtifact, GemmProvenance, Precision, Reproducibility};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::models::Profile;

/// Schema of the binding files.
const BINDING_SCHEMA: &str = "basal-gemm-binding.1";

/// `gemm_table` of a model configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TableRequest {
    Auto,
    Pinned,
    Digest(String),
    None,
    Path(PathBuf),
}

impl TableRequest {
    pub fn parse(value: &str) -> Result<Self> {
        Ok(match value {
            "auto" => Self::Auto,
            "pinned" => Self::Pinned,
            "none" => Self::None,
            v => match v.strip_prefix("sha256:") {
                Some(hex) => {
                    ensure!(
                        hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
                        "gemm_table {v:?}: expected sha256: and 64 lowercase hex digits"
                    );
                    Self::Digest(hex.to_string())
                }
                None => Self::Path(PathBuf::from(v)),
            },
        })
    }

    /// Whether the plan of this request is reproducible (every artifact pinned by digest).
    pub fn reproducibility(&self) -> Reproducibility {
        match self {
            Self::Pinned | Self::Digest(_) => Reproducibility::Pinned,
            _ => Reproducibility::Adaptive,
        }
    }
}

/// The result of the first plan phase for GEMM: the table (if any), whether cuBLASLt tables apply at all on this
/// build and precision, and the reproducibility the request asks for.
#[derive(Clone, Debug)]
pub struct ResolvedGemm {
    pub table: Option<GemmArtifact>,
    pub tables_apply: bool,
    pub reproducibility: Reproducibility,
}

impl ResolvedGemm {
    /// The store file of the table; None without a table or when it was not stored.
    pub fn path(&self) -> Option<&Path> {
        self.table.as_ref().and_then(GemmArtifact::path)
    }

    /// No table and nothing to pin.
    pub fn none(tables_apply: bool) -> Self {
        Self { table: None, tables_apply, reproducibility: Reproducibility::Adaptive }
    }
}

/// Table `t` for messages: its store file and label, or its label when it was not stored.
pub fn describe(t: &GemmArtifact) -> String {
    match t.path() {
        Some(path) => format!("{} ({})", crate::term::P(&path), t.label()),
        None => format!("{} (not stored)", t.label()),
    }
}

/// A table read once: its bytes, the parsed table and the digest of its normalized content.
struct Snapshot {
    text: String,
    value: Value,
    digest: String,
}

impl Snapshot {
    fn parse(text: String, origin: &Path) -> Result<Self> {
        let value: Value =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", crate::term::P(&origin)))?;
        let digest = digest(&value).with_context(|| format!("{}", crate::term::P(&origin)))?;
        Ok(Self { text, value, digest })
    }

    fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", crate::term::P(&path)))?;
        Self::parse(text, path)
    }

    /// The artifact of this snapshot: in the store file `stored` (whose bytes may differ from this snapshot's in
    /// what normalization drops), or only in memory.
    fn artifact(self, stored: Option<StoredFile>, provenance: GemmProvenance) -> GemmArtifact {
        let (path, bytes) = match stored {
            Some(s) => (Some(s.path), s.bytes_sha256),
            None => (None, bytes_sha256(&self.text)),
        };
        GemmArtifact::new(path, bytes, self.digest, provenance, Arc::new(self.value))
    }
}

/// SHA-256 of table bytes (a batch profile's fingerprint records it; the file is never read again for that).
fn bytes_sha256(text: &str) -> [u8; 32] {
    Sha256::digest(text.as_bytes()).into()
}

/// The digest of the normalized content of table `v`.
pub fn digest(v: &Value) -> Result<String> {
    normalize::digest(v).map_err(|e| anyhow::anyhow!("GEMM table: {e}"))
}

/// The bundled table with `digest`: (file name, reduced JSON).
fn bundled(digest: &str) -> Option<(&'static str, &'static str)> {
    crate::config::bundled::TABLES.iter().find(|(_, d, _)| *d == digest).map(|(f, _, t)| (*f, *t))
}

/// Digests of every table of this build.
pub fn bundled_digests() -> impl Iterator<Item = &'static str> {
    crate::config::bundled::TABLES.iter().map(|(_, d, _)| *d)
}

fn artifacts_dir(cache: &Path) -> PathBuf {
    cache.join("artifacts")
}

/// The store file of `digest`.
pub fn artifact_path(cache: &Path, digest: &str) -> PathBuf {
    artifacts_dir(cache).join(format!("{digest}.json"))
}

/// Why the GEMM cache was not written.
enum WriteError {
    /// The cache refuses writes to this process (permission denied, read-only file system): nothing was written.
    Refused(anyhow::Error),
    /// Any other failure: a write that failed (a full disk, an I/O error) or an unreadable or different stored
    /// artifact.
    Failed(anyhow::Error),
}

impl WriteError {
    /// The error of the write `what` (a verb and a path) to the cache.
    fn io(error: std::io::Error, what: String) -> Self {
        use std::io::ErrorKind;
        let refused = matches!(error.kind(), ErrorKind::PermissionDenied | ErrorKind::ReadOnlyFilesystem);
        let error = anyhow::Error::new(error).context(what);
        if refused {
            Self::Refused(error)
        } else {
            Self::Failed(error)
        }
    }

    fn into_inner(self) -> anyhow::Error {
        match self {
            Self::Refused(e) | Self::Failed(e) => e,
        }
    }
}

impl From<anyhow::Error> for WriteError {
    fn from(e: anyhow::Error) -> Self {
        Self::Failed(e)
    }
}

/// A file of the store: its path and the SHA-256 of its bytes.
struct StoredFile {
    path: PathBuf,
    bytes_sha256: [u8; 32],
}

/// A unique temporary file next to `dest` (process, time and a counter), opened for writing.
fn temp_file(dest: &Path) -> Result<(PathBuf, std::fs::File), WriteError> {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = dest.parent().context("artifact path without a directory")?;
    let name = dest.file_name().context("artifact path without a file name")?.to_string_lossy();
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    // Relaxed: the counter only makes names unique within this process.
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = dir.join(format!(".{name}.{}.{nanos}.{n}.tmp", std::process::id()));
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .map_err(|e| WriteError::io(e, format!("creating {}", crate::term::P(&tmp))))?;
    Ok((tmp, file))
}

/// Store table bytes `text` (whose normalized digest is `digest`) in the content-addressed store; an existing file
/// is checked and kept, never replaced. A cache that refuses writes is an error too (`gemm-digest --import`, a
/// search).
fn store(cache: &Path, digest: &str, text: &str) -> Result<StoredFile> {
    try_store(cache, digest, text).map_err(WriteError::into_inner)
}

/// [`store`], telling a refused write apart from every other failure. An existing artifact is found without a
/// write, so a read-only cache that already holds it still yields its file. The store is checked before anything is
/// created: a store that cannot be searched or read is a failure, never a refused write.
fn try_store(cache: &Path, digest: &str, text: &str) -> Result<StoredFile, WriteError> {
    let dest = artifact_path(cache, digest);
    if dest.try_exists().with_context(|| format!("checking {}", crate::term::P(&dest)))? {
        return Ok(check_stored(&dest, digest)?);
    }
    let dir = artifacts_dir(cache);
    std::fs::create_dir_all(&dir).map_err(|e| WriteError::io(e, format!("creating {}", crate::term::P(&dir))))?;
    let (tmp, mut file) = temp_file(&dest)?;
    let written = file.write_all(text.as_bytes()).and_then(|()| file.sync_all());
    drop(file);
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(WriteError::io(e, format!("writing {}", crate::term::P(&tmp))));
    }
    // A hard link fails if the destination exists, so a concurrent writer never replaces a stored artifact.
    let linked = std::fs::hard_link(&tmp, &dest);
    let _ = std::fs::remove_file(&tmp);
    match linked {
        Ok(()) => Ok(StoredFile { path: dest, bytes_sha256: bytes_sha256(text) }),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(check_stored(&dest, digest)?),
        Err(e) => Err(WriteError::io(e, format!("storing {}", crate::term::P(&dest)))),
    }
}

/// The existing store file `path` of `digest`, read once: its normalized digest must be `digest`; its own bytes
/// (not the caller's) are what [`StoredFile::bytes_sha256`] names.
fn check_stored(path: &Path, digest: &str) -> Result<StoredFile> {
    let found = Snapshot::read(path)?;
    ensure!(
        found.digest == digest,
        "{} holds an artifact of sha256 {}, not {digest}; the store is not modified (remove the file to rebuild it)",
        crate::term::P(&path),
        found.digest
    );
    Ok(StoredFile { path: path.to_path_buf(), bytes_sha256: bytes_sha256(&found.text) })
}

/// Store the validated `snapshot` of model `name` for a plan (`what` names its source in messages). A cache that
/// refuses writes leaves it unstored (None, with a warning) and the snapshot is used from memory; any other failure,
/// such as a stored artifact of another content, is an error.
fn store_for_plan(cache: &Path, snapshot: &Snapshot, name: &str, what: &str) -> Result<Option<StoredFile>> {
    match try_store(cache, &snapshot.digest, &snapshot.text) {
        Ok(stored) => Ok(Some(stored)),
        Err(WriteError::Refused(e)) => {
            crate::warn!(
                "{name}: GEMM table {what} (sha256:{}) is used from memory, not stored: the GEMM cache {} is not \
                 writable ({e:#})",
                &snapshot.digest[..16],
                crate::term::P(&cache)
            );
            Ok(None)
        }
        Err(WriteError::Failed(e)) => Err(e),
    }
}

/// A stored artifact by digest, read once and checked against `digest`; None only when it is absent (a store that
/// cannot be searched is an error).
fn stored(cache: &Path, digest: &str) -> Result<Option<(StoredFile, Snapshot)>> {
    let path = artifact_path(cache, digest);
    if !path.try_exists().with_context(|| format!("checking {}", crate::term::P(&path)))? {
        return Ok(None);
    }
    let snapshot = Snapshot::read(&path)?;
    ensure!(
        snapshot.digest == digest,
        "{} holds an artifact of sha256 {}, not {digest}; the store is not modified (remove the file to rebuild it)",
        crate::term::P(&path),
        snapshot.digest
    );
    let bytes_sha256 = bytes_sha256(&snapshot.text);
    Ok(Some((StoredFile { path, bytes_sha256 }, snapshot)))
}

/// What a local binding is for: profile, model, dtype, GPU, cuBLASLt, search version and weight shapes. The profile
/// is part of the key, so a new profile version never inherits the bindings of an earlier one.
struct BindingKey<'a> {
    profile: &'a str,
    model: &'a str,
    dtype: &'a str,
    gpu: &'a str,
    cublaslt: usize,
    shapes: &'a [(usize, usize)],
}

impl BindingKey<'_> {
    fn value(&self) -> Value {
        json!({"profile": self.profile, "model": self.model, "dtype": self.dtype, "gpu": self.gpu,
            "cublaslt_version": self.cublaslt, "gemm_search_version": basal_gpu::gemm_search_version(),
            "shapes": self.shapes})
    }

    fn path(&self, cache: &Path) -> PathBuf {
        let hash = Sha256::digest(self.value().to_string().as_bytes());
        let hex: String = hash.iter().take(16).map(|b| format!("{b:02x}")).collect();
        cache.join("bindings").join(format!("{}-{hex}.json", self.profile))
    }
}

/// The digest bound to `key` on this machine, if the binding is readable and for this key.
fn read_binding(cache: &Path, key: &BindingKey<'_>) -> Option<String> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(key.path(cache)).ok()?).ok()?;
    (v["schema"] == BINDING_SCHEMA && v["for"] == key.value()).then(|| v["sha256"].as_str().map(str::to_string))?
}

/// Bind `digest` to `key`: written to a unique temporary file and renamed over the earlier binding (a binding is
/// an index, the artifacts it names are never replaced).
fn write_binding(cache: &Path, key: &BindingKey<'_>, digest: &str, origin: &str) -> Result<(), WriteError> {
    let path = key.path(cache);
    let dir = path.parent().context("binding path")?;
    std::fs::create_dir_all(dir).map_err(|e| WriteError::io(e, format!("creating {}", crate::term::P(&dir))))?;
    let body = json!({"schema": BINDING_SCHEMA, "for": key.value(), "sha256": digest, "origin": origin});
    let text = serde_json::to_string_pretty(&body).context("serializing a GEMM binding")? + "\n";
    let (tmp, mut file) = temp_file(&path)?;
    let written = file.write_all(text.as_bytes()).and_then(|()| file.sync_all());
    drop(file);
    if let Err(e) = written.and_then(|()| std::fs::rename(&tmp, &path)) {
        let _ = std::fs::remove_file(&tmp);
        return Err(WriteError::io(e, format!("writing {}", crate::term::P(&path))));
    }
    Ok(())
}

/// Provenance of a local artifact: bundled when it is one of this build's tables, else adaptive/local.
fn local_provenance(digest: &str) -> GemmProvenance {
    if bundled_digests().any(|d| d == digest) {
        GemmProvenance::Bundled
    } else {
        GemmProvenance::AdaptiveLocal
    }
}

fn precision_name(p: Precision) -> &'static str {
    match p {
        Precision::F16 => "f16",
        Precision::Bf16 => "bf16",
        Precision::F32 => "f32",
    }
}

/// Why table `v` cannot be pinned for a reproducible plan of `dtype` and the model's weight `shapes` (None: it can):
/// this build's search version exactly, entries of `dtype` only and an entry of `dtype` for every shape.
fn pinned_problem(v: &Value, dtype: &str, shapes: &[(usize, usize)]) -> Option<String> {
    let current = basal_gpu::gemm_search_version();
    let search = v["gemm_search_version"].as_u64().unwrap_or(1);
    let entries = v["entries"].as_array().map_or(&[][..], Vec::as_slice);
    if search != current {
        Some(format!("table of search version {search}; a reproducible plan needs version {current} of this build"))
    } else if let Some(e) = entries.iter().find(|e| e["dtype"].as_str() != Some(dtype)) {
        Some(format!("entry of dtype {} in a reproducible {dtype} plan", e["dtype"]))
    } else if let Some((n, k)) = shapes
        .iter()
        .copied()
        .find(|&(n, k)| !entries.iter().any(|e| e["n"].as_u64() == Some(n as u64) && e["k"].as_u64() == Some(k as u64)))
    {
        Some(format!("no {dtype} entry for shape {n}x{k}"))
    } else {
        None
    }
}

/// Why a backend requiring invariant GEMMs would refuse table `v` in a `dtype` forward: a batch-invariant table
/// with at least one entry, all of `dtype`. Shapes it does not cover keep the fixed per-prompt GEMMs.
fn invariant_problem(v: &Value, dtype: &str) -> Option<String> {
    let entries = v["entries"].as_array().map_or(&[][..], Vec::as_slice);
    if v["invariant"].as_bool() != Some(true) {
        Some("not a batch-invariant table (this backend loads only those)".into())
    } else if entries.is_empty() {
        Some("a table without entries".into())
    } else {
        entries
            .iter()
            .find(|e| e["dtype"].as_str() != Some(dtype))
            .map(|e| format!("entry of dtype {} in a {dtype} forward", e["dtype"]))
    }
}

/// Check table `v` on every GPU of `devices` before any weight is read: cuBLASLt version and GPU name (from the
/// driver, without a context), and with `algorithms` cuBLASLt's acceptance of every algorithm (a batch-invariant
/// table).
fn preflight(v: &Value, version: usize, devices: &[usize], algorithms: bool) -> Result<()> {
    ensure!(
        v["cublaslt_version"].as_u64() == Some(version as u64),
        "table for cuBLASLt {}, this process uses {version}",
        v["cublaslt_version"]
    );
    for &d in devices {
        let gpu = basal_gpu::gpu_identity(d).with_context(|| format!("device {d}"))?;
        ensure!(v["gpu"].as_str() == Some(gpu.name.as_str()), "table for {}, device {d} is {}", v["gpu"], gpu.name);
        if algorithms {
            basal_gpu::check_gemm_table(v, d).with_context(|| format!("device {d}"))?;
        }
    }
    Ok(())
}

/// Phase one of the plan for GEMM: the table `request` (`gemm_table`) of the model of `manifest` with `profile` for
/// the GPUs `devices` of the model (every replica and stage; the first one decides the choice) with precision
/// `precision`, acquired (bundled, stored, imported, searched), read once and checked on every one of the GPUs.
pub fn resolve(
    request: &TableRequest,
    manifest: &ModelManifest,
    profile: &'static Profile,
    precision: Precision,
    cache: &Path,
    devices: &[usize],
) -> Result<ResolvedGemm> {
    let reproducibility = request.reproducibility();
    let pinned = reproducibility == Reproducibility::Pinned;
    let name = &manifest.name;
    let device = *devices.first().with_context(|| format!("{name}: no GPU for the GEMM table"))?;
    if profile.kernel_set == basal_gpu::KernelSet::Qwen35V1 {
        ensure!(
            matches!(request, TableRequest::Auto | TableRequest::None),
            "{name}: Qwen3.5 reference GEMMs support gemm_table auto or none only"
        );
        return Ok(ResolvedGemm { table: None, tables_apply: false, reproducibility });
    }
    // The algorithm search and the tables support half precision only; FP32 uses the reference matmul path, and
    // Metal / Intel have no cuBLASLt: nothing to choose or pin.
    let version = basal_gpu::cublaslt_version().filter(|_| precision != Precision::F32);
    if *request == TableRequest::None {
        return Ok(ResolvedGemm::none(version.is_some()));
    }
    let Some(version) = version else {
        if let TableRequest::Path(p) = request {
            bail!("{name}: GEMM table {} needs the CUDA backend with cuBLASLt and f16 or bf16", crate::term::P(&p));
        }
        return Ok(ResolvedGemm { table: None, tables_apply: false, reproducibility });
    };
    // algorithms are checked on every GPU for a reproducible plan, for a table several GPUs share and for Gemma
    // (whose backend accepts only an invariant table of entries in its compute dtype; partial coverage stays allowed)
    let shared = devices.len() > 1;
    let gemma = matches!(
        manifest.config,
        basal_core::architecture::Architecture::Gemma4(_) | basal_core::architecture::Architecture::Qwen35(_)
    );
    let dtype = precision_name(precision);
    let found = |stored: Option<StoredFile>, snapshot: Snapshot, provenance: GemmProvenance| -> Result<ResolvedGemm> {
        let algorithms = pinned || gemma || (shared && snapshot.value["invariant"].as_bool() == Some(true));
        let context =
            || format!("{name}: GEMM table {}:{} on the model's GPUs", provenance.as_str(), &snapshot.digest[..16]);
        if gemma {
            if let Some(why) = invariant_problem(&snapshot.value, dtype) {
                return Err(anyhow::anyhow!(why)).with_context(context);
            }
        }
        preflight(&snapshot.value, version, devices, algorithms).with_context(context)?;
        Ok(ResolvedGemm { table: Some(snapshot.artifact(stored, provenance)), tables_apply: true, reproducibility })
    };

    if let TableRequest::Path(p) = request {
        // explicit (adaptive): this snapshot, copied into the store if the cache is writable, is what the plan names
        // and the backend loads; the file is not read again
        let snapshot = Snapshot::read(p)?;
        let stored = store_for_plan(cache, &snapshot, name, &crate::term::P(&p).to_string())?;
        if stored.is_some() {
            crate::note!("{name}: GEMM table {} stored as sha256:{}", crate::term::P(&p), &snapshot.digest[..16]);
        }
        return found(stored, snapshot, GemmProvenance::Explicit);
    }
    let gpu = basal_gpu::gpu_name(device)?;
    let shapes = basal_gpu::architecture_projection_shapes(&manifest.config)?;
    let usable = |v: &Value| crate::config::table_problem(v, version, &gpu, dtype, &shapes);
    let pinnable = |v: &Value| usable(v).or_else(|| pinned_problem(v, dtype, &shapes));

    if let TableRequest::Digest(want) = request {
        let (stored, snapshot) = match bundled(want) {
            Some((file, text)) => {
                let snapshot = Snapshot::parse(text.to_string(), Path::new(file))?;
                ensure!(snapshot.digest == *want, "{name}: bundled {file} read as sha256:{}", snapshot.digest);
                (store_for_plan(cache, &snapshot, name, &format!("{file} of this build"))?, snapshot)
            }
            None => {
                let (file, snapshot) = stored(cache, want)?.with_context(|| {
                    format!(
                        "{name}: GEMM artifact sha256:{want} is neither in this build nor in {} (`basal gemm-digest \
                         --import FILE` stores a table)",
                        crate::term::P(&artifacts_dir(cache))
                    )
                })?;
                (Some(file), snapshot)
            }
        };
        ensure!(snapshot.digest == *want, "{name}: GEMM artifact sha256:{want} read as sha256:{}", snapshot.digest);
        if let Some(why) = pinnable(&snapshot.value) {
            bail!("{name}: GEMM artifact sha256:{want} cannot be used here: {why}");
        }
        return found(stored, snapshot, GemmProvenance::Pinned);
    }

    let key = BindingKey { profile: profile.id, model: name, dtype, gpu: &gpu, cublaslt: version, shapes: &shapes };
    if *request == TableRequest::Auto {
        if let Some((stored, snapshot)) = local_table(cache, &key, profile.legacy_cache_import, &usable)? {
            let provenance = local_provenance(&snapshot.digest);
            crate::note!(
                "{name}: GEMM table {}:{} of this machine's cache",
                provenance.as_str(),
                &snapshot.digest[..16]
            );
            return found(stored, snapshot, provenance);
        }
    }
    // the tables of this build the profile pins, in its order, each checked on this GPU
    for pin in profile.gemm_pins() {
        let Some(&(file, built_digest, text)) = crate::config::bundled::TABLES.iter().find(|(f, _, _)| *f == pin.file)
        else {
            // not in this build (another cuBLASLt toolkit, or a build without CUDA tables)
            continue;
        };
        ensure!(
            built_digest == pin.sha256,
            "{name}: profile {} pins {file} as sha256:{}, this build's file is sha256:{built_digest}; a changed table \
             needs a new profile pin",
            profile.id,
            pin.sha256
        );
        let snapshot = Snapshot::parse(text.to_string(), Path::new(file))?;
        ensure!(snapshot.digest == pin.sha256, "{name}: bundled {file} read as sha256:{}", snapshot.digest);
        let problem = if pinned { pinnable(&snapshot.value) } else { usable(&snapshot.value) };
        if problem.is_some() {
            continue;
        }
        match basal_gpu::check_gemm_table(&snapshot.value, device) {
            Ok(()) => {
                let stored = store_for_plan(cache, &snapshot, name, &format!("{file} of this build"))?;
                crate::done!(
                    "{name}: GEMM table for {gpu} from this build ({file}, sha256:{}), no search needed",
                    &pin.sha256[..16]
                );
                return found(stored, snapshot, GemmProvenance::Bundled);
            }
            Err(e) => crate::warn!("{name}: GEMM table {file} of this build rejected ({e:#})"),
        }
    }
    // the bundled tables are built in for one cuBLASLt version: a pin missing from this build must not pass silently
    let built = bundled_digests().count();
    if pinned {
        bail!(
            "{name}: `gemm_table: pinned` and profile {} has no table of this build for {gpu} (cuBLASLt {version}, \
             {dtype}, the model's weight shapes, search version {}; {built} tables built in); use `gemm_table: \
             sha256:<digest>` of a verified table or `auto`",
            profile.id,
            basal_gpu::gemm_search_version()
        );
    }
    if matches!(
        manifest.config,
        basal_core::architecture::Architecture::Gemma4(_) | basal_core::architecture::Architecture::Qwen35(_)
    ) {
        crate::note!(
            "{name}: using fixed per-prompt GEMMs; an invariant table for this GPU can be generated with gemm-search"
        );
        return Ok(ResolvedGemm { table: None, tables_apply: true, reproducibility });
    }
    let (stored, snapshot) = search_and_store(cache, &key, manifest, device)?;
    found(Some(stored), snapshot, GemmProvenance::AdaptiveLocal)
}

/// `auto`: the table bound on this machine, else the file of an earlier cache imported into the store (the source
/// file is left unchanged), when usable here. A bound artifact that is absent or not usable here falls through; one
/// that is present but unreadable or of another content is an error, never replaced by another table. A cache that
/// refuses writes keeps the earlier file's snapshot unstored and unbound (it is imported again at the next start).
fn local_table(
    cache: &Path,
    key: &BindingKey<'_>,
    legacy_import: bool,
    usable: &dyn Fn(&Value) -> Option<String>,
) -> Result<Option<(Option<StoredFile>, Snapshot)>> {
    if let Some(digest) = read_binding(cache, key) {
        let found = stored(cache, &digest).with_context(|| {
            format!(
                "{}: GEMM binding {} names sha256:{digest}, whose stored artifact cannot be read; no other table is \
                 used in its place (remove the artifact or the binding to choose again)",
                key.model,
                crate::term::P(&key.path(cache))
            )
        })?;
        if let Some((file, snapshot)) = found {
            if usable(&snapshot.value).is_none() {
                return Ok(Some((Some(file), snapshot)));
            }
        }
    }
    if !legacy_import {
        return Ok(None);
    }
    let legacy = cache.join(format!("{}-{}.json", key.model, key.dtype));
    let Ok(text) = std::fs::read_to_string(&legacy) else { return Ok(None) };
    let Ok(v) = serde_json::from_str::<Value>(&text) else { return Ok(None) };
    if usable(&v).is_some() {
        return Ok(None);
    }
    let digest = digest(&v).with_context(|| format!("{}", crate::term::P(&legacy)))?;
    let snapshot = Snapshot { text, value: v, digest };
    let what = format!("of the earlier cache {}", crate::term::P(&legacy));
    let Some(file) = store_for_plan(cache, &snapshot, key.model, &what)? else { return Ok(Some((None, snapshot))) };
    match write_binding(cache, key, &snapshot.digest, "imported from an earlier GEMM cache") {
        Ok(()) => {}
        Err(WriteError::Refused(e)) => crate::warn!(
            "{}: GEMM binding to sha256:{} not written, the earlier cache is imported again at the next start ({e:#})",
            key.model,
            &snapshot.digest[..16]
        ),
        Err(WriteError::Failed(e)) => return Err(e),
    }
    crate::note!(
        "{}: GEMM table {what} imported as sha256:{} ({}); the file is left unchanged",
        key.model,
        &snapshot.digest[..16],
        local_provenance(&snapshot.digest).as_str()
    );
    Ok(Some((Some(file), snapshot)))
}

/// Search the batch-invariant table on GPU `device`, store it and bind it to `key`.
fn search_and_store(
    cache: &Path,
    key: &BindingKey<'_>,
    manifest: &ModelManifest,
    device: usize,
) -> Result<(StoredFile, Snapshot)> {
    let classes = crate::config::INVARIANT_CLASSES;
    crate::progress!(
        "{}: generating the batch-invariant GEMM table for {} (device {device}; {} classes x {} shapes, one-time) -> {}",
        manifest.name,
        key.gpu,
        classes.len(),
        key.shapes.len(),
        crate::term::P(&artifacts_dir(cache))
    );
    let t = std::time::Instant::now();
    let mut table = crate::config::search(key.shapes, key.dtype, device)?;
    // for `basal gemm-share`: which build and model made the table
    table["model"] = Value::from(manifest.name.clone());
    table["basal"] = Value::from(format!("{} ({})", env!("CARGO_PKG_VERSION"), env!("BASAL_GIT_SHA")));
    let text = serde_json::to_string_pretty(&table)? + "\n";
    let digest = digest(&table)?;
    let file = store(cache, &digest, &text)?;
    write_binding(cache, key, &digest, "gemm search").map_err(WriteError::into_inner)?;
    crate::done!(
        "{}: GEMM table adaptive/local:{} done in {:.0} s",
        manifest.name,
        &digest[..16],
        t.elapsed().as_secs_f64()
    );
    crate::note!(
        "basal {} has no GEMM table of {} for {}: `basal gemm-share` sends this one to the basal-rs project (a GitHub \
         issue, after your confirmation), so that later builds spare other users of this GPU the search",
        env!("CARGO_PKG_VERSION"),
        manifest.name,
        key.gpu
    );
    Ok((file, Snapshot { text, value: table, digest }))
}

/// `basal gemm-digest`: print the digest of each table file; with `import`, store it in `cache`.
pub fn digest_command(files: &[PathBuf], import: Option<&Path>) -> Result<()> {
    for f in files {
        let text = std::fs::read_to_string(f).with_context(|| format!("reading {}", crate::term::P(&f)))?;
        let v: Value = serde_json::from_str(&text).with_context(|| format!("parsing {}", crate::term::P(&f)))?;
        let d = digest(&v).with_context(|| format!("{}", crate::term::P(&f)))?;
        println!("sha256:{d}  {}", f.display());
        if let Some(cache) = import {
            let file = store(cache, &d, &text)?;
            crate::note!("stored as {}", crate::term::P(&file.path));
        }
    }
    Ok(())
}
