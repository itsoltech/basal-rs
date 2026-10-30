//! CPU planning profiles of the served models and their content fingerprints.
//!
//! A [`ModelProfile`] is captured on the GPU thread right after an engine is built: an offline planner with its own
//! tokenizer (no backend), the backend's host staging bound and background support, `forward_max_tokens` and a
//! numeric profile. The numeric profile is the backend description without device ordinals, the engine batching
//! options and the SHA-256 of the GEMM table. [`ModelProfile::fingerprint`] hashes the model files the loader reads
//! together with that profile and the planner contract version. It runs once at service start on a blocking thread,
//! never in a request. Host paths and device ordinals are not part of the fingerprint.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use basal_core::planner::{HostStaging, RequestPlanner, ShapeLimits};
use basal_core::{Backend, BackgroundSupport, Engine};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::config::BackgroundQualification;
use super::planning::PlanLimits;
use super::store::{Fingerprint, PlannerVersion, WorkerTarget};

/// Contract version of offline planning: bump it whenever prompts, planning or result serialization change, so
/// queued batches are never executed by a different planner.
pub(crate) const PLANNER_VERSION: &str = "basal-planner.5";

/// Files of a model directory that change its results (in this order); missing optional files are recorded as
/// absent. Safetensors shards come from `model.safetensors.index.json` when present.
const MODEL_FILES: [&str; 9] = [
    "config.json",
    "basal.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "evidence_head.pt",
    "model.safetensors",
    "model.safetensors.index.json",
];

/// The planning profile could not be captured or fingerprinted.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ProfileError {
    #[error("model {model}: offline planner")]
    Planner {
        model: String,
        #[source]
        source: basal_core::planner::PlannerLoadError,
    },
    #[error("model {model}: reading {file} for its fingerprint")]
    Read {
        model: String,
        file: String,
        #[source]
        source: std::io::Error,
    },
    #[error("model {model}: {reason}")]
    Invalid { model: String, reason: String },
}

/// Planning profile of one served model, captured from its engine (see the module docs).
pub(crate) struct ModelProfile {
    name: String,
    dir: PathBuf,
    planner: RequestPlanner,
    host_staging: Option<HostStaging>,
    background: BackgroundSupport,
    forward_max_tokens: usize,
    shape_limits: ShapeLimits,
    numeric: Value,
    /// SHA-256 of the bytes of the GEMM table the engine's backend loaded, captured when the table was read or stored
    /// before any weight (`GemmArtifact::bytes_sha256`); the table is never read again from a path.
    gemm_table_sha256: Option<[u8; 32]>,
    qualification: BackgroundQualification,
}

impl ModelProfile {
    /// Capture the profile of `engine` (call on its GPU thread, before the engine moves to a worker). `gemm_table_sha256`
    /// is the SHA-256 of the bytes of the GEMM table the engine's backend uses, if any. `served_name` is the
    /// deployment/catalog identity; the planner and numeric model identity remain those of the source engine.
    pub(crate) fn from_engine<B: Backend>(
        engine: &Engine<B>,
        served_name: &str,
        gemm_table_sha256: Option<&[u8; 32]>,
    ) -> Result<Self, ProfileError> {
        let name = engine.manifest.name.clone();
        let planner = engine.planner().map_err(|source| ProfileError::Planner { model: name.clone(), source })?;
        let mut describe = engine.backend.describe();
        mask_device_ordinals(&mut describe);
        let options = engine.plan_options();
        let mut numeric = serde_json::json!({
            "model": name,
            "version": engine.manifest.version,
            "backend": describe,
            "request_batching": format!("{:?}", engine.request_batching),
            "tree_max_tokens": engine.tree_max_tokens,
            "forward_max_tokens": engine.forward_max_tokens,
            "mark_state": options.mark_state,
            "large_choice": options.large_choice,
            "large_choice_strategy": format!("{:?}", options.large_choice_strategy),
        });
        crate::model_options::record_effective_settings(
            &mut numeric,
            &engine.manifest,
            engine.auto_orders(),
            engine.context_overflow(),
        );
        numeric["thinking"] = engine.thinking_profile();
        numeric["early_exit"] =
            engine.backend.early_exit_caps().map(basal_core::early_exit::profile).unwrap_or(Value::Null);
        mask_device_ordinals(&mut numeric["thinking"]);
        // Only under whole-request isolation; the default fingerprint stays byte-for-byte unchanged.
        if engine.request_isolation() == basal_core::engine::RequestIsolation::Whole {
            numeric["request_isolation"] = "whole".into();
        }
        // Alias-specific routing and result bounds must not share work with the source-name deployment. Keep
        // the old fingerprint byte-for-byte when the served name is unchanged; source provenance stays intact.
        if served_name != name {
            numeric["served_name"] = served_name.into();
        }
        Ok(Self {
            name: served_name.into(),
            dir: engine.manifest.dir.clone(),
            planner,
            host_staging: engine.backend.host_staging(),
            background: engine.background_support(),
            forward_max_tokens: engine.forward_max_tokens,
            shape_limits: engine.shape_limits(),
            numeric,
            gemm_table_sha256: gemm_table_sha256.copied(),
            qualification: BackgroundQualification::not_configured(),
        })
    }

    /// Attach the operator's background qualification (default: disabled).
    pub(crate) fn with_background(mut self, qualification: BackgroundQualification) -> Self {
        self.qualification = qualification;
        self
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// Hash the model files, the numeric profile, the planner version and the planning admission limits. Blocking
    /// (reads every weight file once).
    ///
    /// The admission limits (`R`, `P`, line, record and JSON limits) are part of the fingerprint. Two hosts with one
    /// fingerprint therefore validate every record identically and agree on `max_record_cost` and `host_bound ≤ P`,
    /// so neither can accept work the other must refuse. Changing a host budget needs a new catalog revision, and the
    /// old hosts drain their batches (conservative by design).
    pub(crate) fn resolve(mut self, limits: &PlanLimits) -> Result<ResolvedProfile, ProfileError> {
        let exec_pool_bytes = limits.pool_bytes;
        // DRR costs are nanoseconds of device-gate time: (base packed tokens + thinking work) × the qualified
        // coefficient (1 without a qualification, whose profile does not execute). A new coefficient is a new fingerprint.
        // Keep the measured memory and composition contract in the profile identity. Local shared-device
        // admission can be stricter and defers work; it cannot permanently reject a compatible record.
        // The evidence reference is a label and is left out.
        let (cost_ns_per_token, background) = match &self.qualification {
            BackgroundQualification::Qualified {
                device_bytes,
                online_reserve_bytes,
                allocation_granularity_bytes,
                cost_ns_per_token,
                composition,
                max_requests,
                ..
            } => (
                *cost_ns_per_token,
                serde_json::json!({
                    "device_bytes": device_bytes,
                    "online_reserve_bytes": online_reserve_bytes,
                    "allocation_granularity_bytes": allocation_granularity_bytes,
                    "composition": format!("{composition:?}"),
                    "max_requests": max_requests,
                }),
            ),
            BackgroundQualification::Disabled(_) => (1, serde_json::Value::String("disabled".into())),
        };
        self.numeric["admission"] = serde_json::json!({
            "background": background,
            "cost_ns_per_token": cost_ns_per_token,
            "plan_slot_bytes": limits.slot_bytes,
            "exec_pool_bytes": limits.pool_bytes,
            "max_line_bytes": limits.input.max_line_bytes,
            "max_records": limits.input.max_records,
            "max_json_depth": limits.input.max_json_depth,
            "max_json_nodes": limits.input.max_json_nodes,
        });
        let fingerprint = self.fingerprint()?;
        let planner_version = PlannerVersion::parse(PLANNER_VERSION).map_err(|e| self.invalid(e.to_string()))?;
        // Inherited DRR policy cap, also applied to total thinking work; this is not a thinking host-byte bound.
        // P ≤ u32::MAX and the coefficient ≤ 1e9 keep this below i64::MAX / 2.
        let max_record_cost = match &self.host_staging {
            Some(staging) => (exec_pool_bytes / staging.per_token.max(1))
                .checked_mul(cost_ns_per_token)
                .ok_or_else(|| self.invalid("max_record_cost overflows"))?,
            None => 0,
        };
        Ok(ResolvedProfile {
            name: self.name,
            fingerprint,
            planner_version,
            planner: Arc::new(self.planner),
            host_staging: self.host_staging,
            background: self.background,
            forward_max_tokens: self.forward_max_tokens,
            shape_limits: self.shape_limits,
            qualification: self.qualification,
            max_record_cost,
            cost_ns_per_token,
        })
    }

    fn fingerprint(&self) -> Result<Fingerprint, ProfileError> {
        let read = |file: &str, source| ProfileError::Read { model: self.name.clone(), file: file.to_string(), source };
        let mut h = Sha256::new();
        let mut field = |bytes: &[u8]| {
            h.update((bytes.len() as u64).to_be_bytes());
            h.update(bytes);
        };
        field(b"basal.batch.fingerprint.v1");
        field(PLANNER_VERSION.as_bytes());
        field(&serde_json::to_vec(&self.numeric).map_err(|e| self.invalid(format!("numeric profile: {e}")))?);
        match &self.gemm_table_sha256 {
            Some(table) => field(table),
            None => field(b"no-gemm-table"),
        }
        let mut files: Vec<String> = MODEL_FILES.iter().map(|f| (*f).to_string()).collect();
        files.extend(self.calibration_files().map_err(|e| read("CALIBRATION*.json", e))?);
        files.extend(self.shards().map_err(|e| read("model.safetensors.index.json", e))?);
        for name in files {
            let path = self.dir.join(&name);
            field(name.as_bytes());
            match std::fs::metadata(&path) {
                Ok(meta) if meta.is_file() => field(&file_digest(&path).map_err(|e| read(&name, e))?),
                Ok(_) => field(b"not-a-file"),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => field(b"absent"),
                Err(e) => return Err(read(&name, e)),
            }
        }
        let digest = h.finalize();
        Fingerprint::parse(&format!("fp1-{}", super::config::hex(&digest))).map_err(|e| self.invalid(e.to_string()))
    }

    fn invalid(&self, reason: impl Into<String>) -> ProfileError {
        ProfileError::Invalid { model: self.name.clone(), reason: reason.into() }
    }

    /// `CALIBRATION.json` and its per-order variants, sorted.
    fn calibration_files(&self) -> std::io::Result<Vec<String>> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let name = entry?.file_name();
            if let Some(name) = name.to_str().filter(|n| n.starts_with("CALIBRATION") && n.ends_with(".json")) {
                names.push(name.to_string());
            }
        }
        names.sort();
        Ok(names)
    }

    /// Shard files named by the safetensors index, sorted and deduplicated; none without an index.
    fn shards(&self) -> std::io::Result<Vec<String>> {
        let index = self.dir.join("model.safetensors.index.json");
        let text = match std::fs::read(&index) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let v: Value = serde_json::from_slice(&text).map_err(std::io::Error::other)?;
        let mut shards: Vec<String> = v
            .get("weight_map")
            .and_then(Value::as_object)
            .map(|m| m.values().filter_map(Value::as_str).map(str::to_string).collect())
            .unwrap_or_default();
        shards.sort();
        shards.dedup();
        if shards.iter().any(|s| s.contains('/') || s.contains('\\') || s.starts_with('.')) {
            return Err(std::io::Error::other("a shard name is not a plain file name"));
        }
        Ok(shards)
    }
}

/// Remove device ordinals from a backend description, recursively, keeping everything else (layer ranges, numeric
/// modes, hardware types): `device`/`devices` keys with numeric values are dropped, and `Device::new_cuda(N)` /
/// `Device::new_metal(N)` in text become `(N)`.
fn mask_device_ordinals(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.retain(|key, v| {
                let ordinal = v.is_number() || v.as_array().is_some_and(|a| a.iter().all(Value::is_number));
                !(matches!(key.as_str(), "device" | "devices" | "ordinal" | "ordinals") && ordinal)
            });
            map.values_mut().for_each(mask_device_ordinals);
        }
        Value::Array(items) => items.iter_mut().for_each(mask_device_ordinals),
        Value::String(text) => {
            for call in ["new_cuda(", "new_metal("] {
                let mut out = String::with_capacity(text.len());
                let mut rest = text.as_str();
                while let Some(i) = rest.find(call) {
                    let after = &rest[i + call.len()..];
                    let digits = after.bytes().take_while(u8::is_ascii_digit).count();
                    out.push_str(&rest[..i + call.len()]);
                    out.push_str(if digits > 0 { "N" } else { "" });
                    rest = &after[digits..];
                }
                out.push_str(rest);
                *text = out;
            }
        }
        _ => {}
    }
}

/// SHA-256 of a file's content, read in 1 MiB blocks (follows symlinks: Hugging Face snapshots link to blobs).
fn file_digest(path: &Path) -> std::io::Result<[u8; 32]> {
    let mut file = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().into())
}

/// A fingerprinted profile shared by the validator, the planning threads and the background executor.
pub(crate) struct ResolvedProfile {
    /// Model name (the catalog's `model_name`).
    pub(crate) name: String,
    pub(crate) fingerprint: Fingerprint,
    pub(crate) planner_version: PlannerVersion,
    pub(crate) planner: Arc<RequestPlanner>,
    /// `None`: no host staging bound, so the profile neither validates nor executes batches.
    pub(crate) host_staging: Option<HostStaging>,
    pub(crate) background: BackgroundSupport,
    pub(crate) forward_max_tokens: usize,
    /// Engine limits of the device work shape of a plan (`DeviceWorkShape::from_plans`).
    pub(crate) shape_limits: ShapeLimits,
    pub(crate) qualification: BackgroundQualification,
    /// Highest admissible record cost in ns: `P / max(1, host_staging.per_token) × cost_ns_per_token`; validation
    /// refuses records above it.
    pub(crate) max_record_cost: u64,
    /// DRR cost coefficient (ns of device-gate time per token of base packed/thinking work); 1 if unqualified.
    pub(crate) cost_ns_per_token: u64,
}

impl ResolvedProfile {
    /// The validation target of this profile, when it can plan bounded records (health `validates`).
    pub(crate) fn target(&self) -> Option<WorkerTarget> {
        self.host_staging.is_some().then(|| self.claim_target())
    }

    /// The claim target of this profile whether or not it can plan; validation refuses an unbounded profile.
    pub(crate) fn claim_target(&self) -> WorkerTarget {
        WorkerTarget { fingerprint: self.fingerprint.clone(), planner_version: self.planner_version.clone() }
    }
}

impl std::fmt::Debug for ResolvedProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedProfile")
            .field("name", &self.name)
            .field("fingerprint", &self.fingerprint)
            .field("max_record_cost", &self.max_record_cost)
            .finish_non_exhaustive()
    }
}
