//! Trained exit capabilities and request-isolated upstream units/chunks. No global request policy mutation.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::pack::{pack_exit, Packed, BATCHES, TOKEN_BUDGET};
use crate::request::{DecideError, Dialect, FieldError};

pub const EXIT_REPO: &str = "Remek/basal-1.0-4.5B";
pub const EXIT_REVISION: &str = "b95288041cb1975bb930c6f0410819273fd2f54f";
pub const WEIGHTS_SHA256: &str = "3cfe36727cbacc54b3f85f382319c877a3fa07689591cbf661ec5e90dbf8f0f3";
pub const HEADS_SHA256: &str = "4e6752762132ffe97f2d21cfe63ffd5f59b281451d583c989fe5c65cff49eb65";
pub const THRESHOLDS_SHA256: &str = "5cf1b71833ce85bb5ea39ac5b02fb7c84f7b20c52a78810b4ede9f36430fc6bc";
pub const GROUPING: &str = "request-isolated upstream units/chunks; exit-grid-v1; budget=12288; cache=bypass";
pub const LENS: [usize; 18] =
    [128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 3072];

/// Validated published artifacts, before any device activity. Files must stay unchanged throughout model loading.
/// No acquisition happens here. The private fields prevent enabling heads from dimensionality alone.
#[derive(Clone, Debug)]
pub struct ExitArtifacts {
    model_dir: PathBuf,
    dir: PathBuf,
    caps: ExitCaps,
}

impl ExitArtifacts {
    pub fn load(model: &crate::ModelManifest, heads_dir: Option<&Path>, default: &str) -> Result<Self> {
        use sha2::{Digest, Sha256};

        use crate::prompt::{ChatStyle, PromptFormat};
        ensure!(
            model.name == "basal-1.0-4.5B"
                && model.chat_style == ChatStyle::Llama
                && model.prompt_format == PromptFormat::One,
            "trained exits require {EXIT_REPO}@{EXIT_REVISION}"
        );
        let cfg = model.config.llama()?;
        let dir = heads_dir.map(Path::to_path_buf).unwrap_or_else(|| model.dir.join("exit_heads"));
        for (path, expected) in [
            (model.dir.join("config.json"), "7ee4821083714faa535d582a06a1756cb8ee1db36ee7d75299476e30b44b9aa2"),
            (model.dir.join("basal.json"), "d9a60ef12e0f05607e5266d9d619d7c6d5f47017068655fd7c53613b0f09bfc1"),
            (model.dir.join("model.safetensors"), WEIGHTS_SHA256),
            (dir.join("heads.pt"), HEADS_SHA256),
        ] {
            ensure!(
                file_hash(&path)? == expected,
                "trained exits: {} differs from the published artifact",
                path.display()
            );
        }
        let thresholds_path = dir.join("thresholds.json");
        let thresholds =
            std::fs::read(&thresholds_path).with_context(|| format!("reading {}", thresholds_path.display()))?;
        ensure!(
            format!("{:x}", Sha256::digest(&thresholds)) == THRESHOLDS_SHA256,
            "trained exits: {} differs from the published artifact",
            thresholds_path.display()
        );
        let caps = ExitCaps::from_thresholds(&thresholds, default, cfg.num_layers)?;
        Ok(Self { model_dir: model.dir.canonicalize()?, dir, caps })
    }

    pub fn check_model_dir(&self, dir: &Path) -> Result<()> {
        ensure!(dir.canonicalize()? == self.model_dir, "trained artifacts belong to another model directory");
        Ok(())
    }

    pub fn heads_path(&self) -> PathBuf {
        self.dir.join("heads.pt")
    }
    pub fn caps(&self) -> &ExitCaps {
        &self.caps
    }
}

fn file_hash(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Reproducible identity without local paths; runtime tensor inventory is separately loader-validated.
pub fn profile(caps: &ExitCaps) -> Value {
    serde_json::json!({"repo":EXIT_REPO,"revision":EXIT_REVISION,
        "weights_sha256":WEIGHTS_SHA256,"heads_sha256":HEADS_SHA256,"thresholds_sha256":THRESHOLDS_SHA256,
        "default":caps.default_policy(),"available":caps.available(),"grouping":GROUPING,
        "qualification":"unverified runtime/numerics; policy labels are fitted agreement targets"})
}

/// Immutable CPU capability snapshot shared by online and offline planners. Construction validates the inventory.
#[derive(Clone, Debug, Serialize)]
pub struct ExitCaps {
    policies: BTreeMap<String, BTreeMap<usize, f64>>,
    default: String,
}

impl ExitCaps {
    /// Parse the artifact's policy inventory; disabled (>1) thresholds remain validated but are not executed.
    pub fn from_thresholds(bytes: &[u8], default: &str, layers: usize) -> Result<Self> {
        #[derive(Deserialize)]
        struct Level {
            taus: BTreeMap<usize, f64>,
        }
        #[derive(Deserialize)]
        struct Thresholds {
            levels: BTreeMap<String, Level>,
        }
        let source: Thresholds = serde_json::from_slice(bytes).context("exit thresholds")?;
        ensure!(!source.levels.is_empty(), "exit thresholds contain no policies");
        let mut policies = BTreeMap::new();
        for (name, level) in source.levels {
            ensure!(!name.is_empty() && name != "off", "invalid exit policy name");
            ensure!(!level.taus.is_empty(), "exit policy {name} has no thresholds");
            for (&layer, &tau) in &level.taus {
                ensure!(layer > 0 && layer < layers, "exit layer {layer} outside intermediate decoder range");
                ensure!(tau.is_finite() && tau >= 0.0, "exit layer {layer}: invalid threshold");
            }
            policies.insert(name, level.taus);
        }
        policies.insert("off".into(), BTreeMap::new());
        ensure!(policies.contains_key(default), "unknown default early_exit {default:?}");
        Ok(Self { policies, default: default.into() })
    }

    pub fn available(&self) -> Vec<&str> {
        self.policies.keys().map(String::as_str).collect()
    }
    pub fn default_policy(&self) -> &str {
        &self.default
    }
    pub fn thresholds(&self, policy: &str) -> Result<&BTreeMap<usize, f64>> {
        self.policies.get(policy).with_context(|| format!("unavailable early_exit policy {policy:?}"))
    }
    pub fn layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.policies.values().flat_map(|p| p.keys().copied())
    }
}

/// Missing/null inherits the backend default. Plain backends refuse every explicit non-null value, including off.
pub(crate) fn request_policy(
    body: &Value,
    caps: Option<&ExitCaps>,
    dialect: Dialect,
) -> Result<Option<String>, DecideError> {
    let Some(value) = body.get("early_exit").filter(|v| !v.is_null()) else {
        return Ok(caps.map(|c| c.default_policy().into()));
    };
    let error = |message: String, kind| vec![FieldError::new(vec!["body".into(), "early_exit".into()], message, kind)];
    let Some(caps) = caps else {
        return Err(DecideError::Unsupported(error(
            "early_exit requires an enabled trained-head backend".into(),
            "unsupported_feature",
        )));
    };
    if ["media", "assets"].iter().any(|f| body.get(*f).is_some_and(|v| !v.is_null())) {
        return Err(DecideError::Unsupported(error(
            "media does not support explicit early_exit".into(),
            "unsupported_feature",
        )));
    }
    let policy = match value {
        Value::String(s) => s.clone(),
        Value::Number(n) if dialect != Dialect::TypeSafe => {
            n.as_f64().filter(|v| v.is_finite()).map(|v| v.to_string()).ok_or_else(|| {
                DecideError::Validation(error(
                    "early_exit number must be finite and representable as f64".into(),
                    "value_error",
                ))
            })?
        }
        _ if dialect != Dialect::TypeSafe => value.to_string(),
        _ => return Err(DecideError::Validation(error("early_exit must be a string or null".into(), "value_error"))),
    };
    if caps.thresholds(&policy).is_err() {
        return Err(DecideError::Validation(error(
            format!("unavailable early_exit {policy:?}; available: {:?}", caps.available()),
            "value_error",
        )));
    }
    Ok(Some(policy))
}

pub(crate) struct Unit {
    pub packed: Packed,
    /// (prepared question, option order), retained through every bisection and length sort.
    pub owners: Vec<(usize, usize)>,
}

/// One request group, all question/order/label readouts. Upstream _units bisects readouts, not questions.
pub(crate) fn units(prepared: &[crate::Prepared]) -> Result<Vec<Unit>> {
    let owners: Vec<_> =
        prepared.iter().enumerate().flat_map(|(i, p)| (0..p.orders.len()).map(move |o| (i, o))).collect();
    if owners.is_empty() {
        return Ok(Vec::new());
    }
    let mut todo = vec![(0, owners.len())];
    let mut out = Vec::new();
    while let Some((a, b)) = todo.pop() {
        let tokens: Vec<_> = owners[a..b].iter().map(|&(i, o)| prepared[i].orders[o].input_ids.as_slice()).collect();
        let packed = pack_exit(&tokens)?;
        let root = packed.node.iter().position(|&n| n == 1).unwrap_or(packed.ids.len());
        let oversize = packed.parent.iter().filter(|&&p| p == -1).count() > 1 || packed.ids.len() - root > 8192;
        if oversize && b - a > 1 {
            let mid = (a + b) / 2;
            todo.extend([(mid, b), (a, mid)]);
        } else {
            out.push(Unit { packed, owners: owners[a..b].to_vec() });
        }
    }
    Ok(out)
}

pub fn length_bucket(n: usize) -> Option<usize> {
    LENS.into_iter().find(|&x| n <= x)
}
pub fn batch_bucket(n: usize) -> Option<usize> {
    BATCHES.into_iter().find(|&x| n <= x)
}

pub(crate) fn chunks(units: &[Unit]) -> Vec<Vec<usize>> {
    let mut order: Vec<_> = (0..units.len()).collect();
    order.sort_by_key(|&i| units[i].packed.ids.len());
    let mut chunks = Vec::new();
    let mut current = Vec::new();
    for i in order {
        let fits = batch_bucket(current.len() + 1)
            .zip(length_bucket(units[i].packed.ids.len()))
            .is_some_and(|(b, l)| b * l <= TOKEN_BUDGET);
        if !current.is_empty() && !fits {
            chunks.push(std::mem::take(&mut current));
        }
        current.push(i);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}
