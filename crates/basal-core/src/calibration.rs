//! Calibration file selection and explicit-file identity checks, independent of the serving backend.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::manifest::ModelLoadOptions;
use crate::prompt::{OptionOrders, PromptLayout};
use crate::weights::{CalibrationCapture, CalibrationProbeRole, CalibrationReadRole, OrdersVariantDecision};

#[derive(Clone, Copy, Debug, Default)]
pub enum CalibrationSource<'a> {
    #[default]
    Automatic,
    File(&'a Path),
    Off,
}

/// Identity of the actual served model, supplied by the application rather than guessed from basal.json.
#[derive(Clone, Copy, Debug)]
pub struct ServedIdentity<'a> {
    pub model: &'a str,
    pub revision: Option<&'a str>,
}

#[derive(Clone, Debug)]
pub struct CalibrationProvenance {
    pub path: Option<PathBuf>,
    /// SHA-256 of the exact file bytes; absent for off or a missing automatic file.
    pub content_sha256: Option<String>,
    pub explicit: bool,
    pub off: bool,
    pub strict: bool,
    pub identity_warning: Option<String>,
    fitted_model: Option<String>,
    fitted_revision: Option<String>,
}

#[derive(Debug, thiserror::Error)]
#[error("explicit calibration identity mismatch: {0} (--strict-calibration)")]
pub struct CalibrationIdentityError(pub String);

impl CalibrationProvenance {
    /// Only explicit files require identity checks. A warning preserves temperatures and certificates;
    /// strict mode refuses the file. Call again with the configured identity if it was not known at load time.
    pub fn check_identity(&self, served: ServedIdentity<'_>) -> Result<Option<String>, CalibrationIdentityError> {
        let warning = if self.explicit && !self.off {
            calibration_identity_mismatch(self.fitted_model.as_deref(), self.fitted_revision.as_deref(), served)
        } else {
            None
        };
        if self.strict {
            if let Some(why) = &warning {
                return Err(CalibrationIdentityError(why.clone()));
            }
        }
        Ok(warning)
    }
}

/// Upstream identity matching: compare the last path/repository component without case or trailing slashes;
/// compare revisions only when both are known. No filesystem or model-directory inference is performed.
pub fn calibration_identity_mismatch(
    fitted_model: Option<&str>,
    fitted_revision: Option<&str>,
    served: ServedIdentity<'_>,
) -> Option<String> {
    let fitted = fitted_model.filter(|s| !s.is_empty())?;
    let key = |s: &str| s.trim_end_matches('/').rsplit('/').next().unwrap_or("").to_lowercase();
    if key(fitted) != key(served.model) {
        return Some(format!("it was fitted on {fitted}, the served model is {}", served.model));
    }
    if let (Some(fitted), Some(actual)) = (fitted_revision.filter(|s| !s.is_empty()), served.revision) {
        if !actual.is_empty() && fitted != actual {
            return Some(format!(
                "it was fitted on revision {} of {}, the served revision is {}",
                fitted.chars().take(12).collect::<String>(),
                served.model,
                actual.chars().take(12).collect::<String>()
            ));
        }
    }
    None
}

fn read(path: &Path) -> Result<(Value, String)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let data = serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    Ok((data, format!("{:x}", Sha256::digest(&bytes))))
}

fn fitted_layout(data: &Value) -> Result<&str> {
    let set = |v: &&Value| !v.is_null() && v.as_str() != Some("");
    data.pointer("/fitted_on/layout")
        .filter(set)
        .or_else(|| data.get("layout").filter(set))
        .map_or(Ok("canonical"), |v| v.as_str().context("calibration layout must be a string"))
}

/// Why `data` read from `path` does not hold for the served `layout`; `None` for an empty file or a matching layout.
fn layout_mismatch(data: &Value, path: &Path, layout: PromptLayout) -> Result<Option<String>> {
    if data.as_object().is_none_or(serde_json::Map::is_empty) {
        return Ok(None);
    }
    let fitted = fitted_layout(data)?;
    Ok((fitted != layout.as_str()).then(|| {
        // The file name only (as upstream): the message may reach API metadata, a full local path must not.
        let name = path.file_name().unwrap_or(path.as_os_str()).to_string_lossy();
        format!("{name} was fitted on the {fitted} layout, served with --layout {layout}")
    }))
}

fn fitted_orders(data: &Value) -> Result<Option<OptionOrders>> {
    let set = |v: &&Value| !v.is_null();
    data.pointer("/fitted_on/orders")
        .filter(set)
        .or_else(|| data.get("orders").filter(set))
        .map(|v| OptionOrders::parse(&v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string())))
        .transpose()
}

pub(crate) fn load(
    dir: &Path,
    source: CalibrationSource<'_>,
    layout: PromptLayout,
    orders: OptionOrders,
    strict: bool,
    identity: Option<ServedIdentity<'_>>,
) -> Result<(Value, CalibrationProvenance, Option<String>)> {
    let loaded = load_impl(dir, source, layout, orders, strict, identity, ReadMode::Legacy)?;
    Ok((loaded.data, loaded.provenance, loaded.unvalidated))
}

pub(crate) struct LoadedCalibration {
    pub(crate) data: Value,
    pub(crate) provenance: CalibrationProvenance,
    pub(crate) unvalidated: Option<String>,
}

pub(crate) fn load_captured(
    dir: &Path,
    options: ModelLoadOptions<'_>,
    layout: PromptLayout,
    orders: OptionOrders,
    capture: &mut CalibrationCapture,
) -> Result<LoadedCalibration> {
    load_impl(
        dir,
        options.calibration,
        layout,
        orders,
        options.strict_calibration,
        options.served_identity,
        ReadMode::Captured(capture),
    )
}

enum ReadMode<'a> {
    Legacy,
    Captured(&'a mut CalibrationCapture),
}

impl ReadMode<'_> {
    fn exists(&mut self, path: &Path, role: CalibrationProbeRole) -> bool {
        let exists = path.exists();
        if let Self::Captured(capture) = self {
            capture.record_probe(role, exists);
        }
        exists
    }

    fn read(&mut self, path: &Path, role: CalibrationReadRole) -> Result<(Value, String)> {
        match self {
            Self::Legacy => read(path),
            Self::Captured(capture) => capture.read(path, role),
        }
    }

    fn layout(&mut self, initial: bool, mismatch: bool) {
        if let Self::Captured(capture) = self {
            capture.record_layout(initial, mismatch);
        }
    }

    fn orders(&mut self, initial: bool, fitted: Option<OptionOrders>, served: OptionOrders) {
        if let Self::Captured(capture) = self {
            capture.record_orders(initial, fitted, served);
        }
    }

    fn variant(&mut self, decision: OrdersVariantDecision) {
        if let Self::Captured(capture) = self {
            capture.record_variant(decision);
        }
    }
}

fn load_impl(
    dir: &Path,
    source: CalibrationSource<'_>,
    layout: PromptLayout,
    orders: OptionOrders,
    strict: bool,
    identity: Option<ServedIdentity<'_>>,
    mut mode: ReadMode<'_>,
) -> Result<LoadedCalibration> {
    let explicit = matches!(source, CalibrationSource::File(_));
    let off = matches!(source, CalibrationSource::Off);
    let mut path = match source {
        CalibrationSource::File(path) => path.to_path_buf(),
        _ => {
            let candidate = dir.join(format!("CALIBRATION.{layout}.json"));
            if layout != PromptLayout::Canonical && mode.exists(&candidate, CalibrationProbeRole::LayoutPreference) {
                candidate
            } else {
                dir.join("CALIBRATION.json")
            }
        }
    };
    let (mut data, mut hash) = if !off && (explicit || mode.exists(&path, CalibrationProbeRole::Candidate)) {
        let role = if explicit { CalibrationReadRole::Explicit } else { CalibrationReadRole::Candidate };
        let (data, hash) = mode.read(&path, role)?;
        (data, Some(hash))
    } else {
        (serde_json::json!({}), None)
    };
    // Layout is checked before orders: a mismatch must not be rescued by an orders variant.
    let mut unvalidated = layout_mismatch(&data, &path, layout)?;
    mode.layout(true, unvalidated.is_some());
    if unvalidated.is_none() {
        let replace = if !explicit && !off {
            let fitted = fitted_orders(&data)?;
            mode.orders(true, fitted, orders);
            fitted.is_some_and(|o| o != orders)
        } else {
            false
        };
        if replace {
            // Automatic names are CALIBRATION[.<layout>].json; the variant is CALIBRATION[.<layout>].o<orders>.json.
            let alt = path.with_extension(format!("o{orders}.json"));
            if mode.exists(&alt, CalibrationProbeRole::OrdersVariant) {
                let (value, digest) = mode.read(&alt, CalibrationReadRole::OrdersVariant)?;
                // Validate the replacement too, so an incorrectly packaged orders file cannot certify another layout.
                unvalidated = layout_mismatch(&value, &alt, layout)?;
                mode.layout(false, unvalidated.is_some());
                mode.variant(OrdersVariantDecision::Selected);
                data = value;
                hash = Some(digest);
                path = alt;
            } else {
                mode.variant(OrdersVariantDecision::ProbeFalse);
            }
        }
        if unvalidated.is_none() {
            let fitted = fitted_orders(&data)?;
            mode.orders(false, fitted, orders);
            unvalidated = fitted
                .filter(|o| *o != orders)
                .map(|fitted| format!("calibration was fitted on {fitted} option order(s), served with {orders}"));
        }
    }
    let mut provenance = CalibrationProvenance {
        path: (!off).then_some(path),
        content_sha256: hash,
        explicit,
        off,
        strict,
        identity_warning: None,
        fitted_model: data.pointer("/fitted_on/model").and_then(Value::as_str).map(str::to_owned),
        fitted_revision: data.pointer("/fitted_on/revision").and_then(Value::as_str).map(str::to_owned),
    };
    if let Some(identity) = identity {
        provenance.identity_warning = provenance.check_identity(identity)?;
    }
    Ok(LoadedCalibration { data, provenance, unvalidated })
}
