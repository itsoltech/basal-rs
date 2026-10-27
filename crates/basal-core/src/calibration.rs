//! Calibration file selection and explicit-file identity checks, independent of the serving backend.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::prompt::{OptionOrders, PromptLayout};

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
    data.pointer("/fitted_on/layout")
        .filter(|v| !v.is_null() && v.as_str() != Some(""))
        .or_else(|| data.get("layout").filter(|v| !v.is_null() && v.as_str() != Some("")))
        .map(|v| v.as_str().context("calibration layout must be a string"))
        .transpose()
        .map(|layout| layout.unwrap_or("canonical"))
}

fn fitted_orders(data: &Value) -> Result<Option<OptionOrders>> {
    data.pointer("/fitted_on/orders")
        .filter(|v| !v.is_null())
        .or_else(|| data.get("orders"))
        .filter(|v| !v.is_null())
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
    let explicit = matches!(source, CalibrationSource::File(_));
    let off = matches!(source, CalibrationSource::Off);
    let mut path = match source {
        CalibrationSource::File(path) => path.to_path_buf(),
        _ => {
            let candidate = dir.join(format!("CALIBRATION.{layout}.json"));
            if layout != PromptLayout::Canonical && candidate.exists() {
                candidate
            } else {
                dir.join("CALIBRATION.json")
            }
        }
    };
    let (mut data, mut hash) = if !off && (explicit || path.exists()) {
        let (data, hash) = read(&path)?;
        (data, Some(hash))
    } else {
        (serde_json::json!({}), None)
    };
    // Layout is checked before orders: a mismatch must not be rescued by an orders variant.
    let mut unvalidated = None;
    if data.as_object().is_some_and(|m| !m.is_empty()) && fitted_layout(&data)? != layout.as_str() {
        unvalidated = Some(format!(
            "{} was fitted on the {} layout, served with --layout {layout}",
            path.display(),
            fitted_layout(&data)?
        ));
    } else {
        if fitted_orders(&data)?.is_some_and(|o| o != orders) && !explicit && !off {
            let stem = path.file_stem().and_then(|s| s.to_str()).context("invalid calibration filename")?;
            let alt = path.with_file_name(format!("{stem}.o{orders}.json"));
            if alt.exists() {
                let (value, digest) = read(&alt)?;
                data = value;
                hash = Some(digest);
                path = alt;
            }
        }
        // Validate the replacement too, so an incorrectly packaged orders file cannot certify another layout.
        if data.as_object().is_some_and(|m| !m.is_empty()) && fitted_layout(&data)? != layout.as_str() {
            unvalidated =
                Some(format!("{} was fitted on another layout, served with --layout {layout}", path.display()));
        } else if let Some(fitted) = fitted_orders(&data)?.filter(|o| *o != orders) {
            unvalidated = Some(format!("calibration was fitted on {fitted} option order(s), served with {orders}"));
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
    Ok((data, provenance, unvalidated))
}
