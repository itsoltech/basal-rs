//! Prompt, calibration and adaptive-order options shared by CLI loading and per-model YAML.

use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use basal_core::calibration::{CalibrationSource, ServedIdentity};
use basal_core::engine::AutoOrderSettings;
use basal_core::manifest::ModelLoadOptions;
use basal_core::prompt::PromptLayout;
use basal_core::ModelManifest;
use clap::Args;
use serde::Deserialize;

#[derive(Args, Clone, Default, Deserialize)]
pub(crate) struct ModelOptions {
    /// Prompt layout (otherwise the model declaration, default canonical)
    #[arg(long, value_parser = ["canonical", "qfirst", "state_twice"])]
    #[serde(default)]
    pub layout: Option<String>,
    /// Explicit calibration file; otherwise select the model's layout/order calibration automatically
    #[arg(long, conflicts_with = "no_calibration")]
    #[serde(default)]
    pub calibration: Option<PathBuf>,
    /// Disable calibration, including automatic calibration files
    #[arg(long, conflicts_with = "calibration")]
    #[serde(default)]
    pub no_calibration: bool,
    /// Refuse explicit calibration fitted on another model or known HF revision (otherwise warn)
    #[arg(long)]
    #[serde(default)]
    pub strict_calibration: bool,
    /// Adaptive orders: second pass when top probability is below this threshold (default 0.6)
    #[arg(long)]
    #[serde(default)]
    pub auto_tau: Option<f64>,
    /// Adaptive noul orders: inclusive uncertainty band (default 0.2 0.8)
    #[arg(long, num_args = 2, value_names = ["LO", "HI"])]
    #[serde(default)]
    pub auto_band: Option<Vec<f64>>,
}

impl ModelOptions {
    pub(crate) fn is_set(&self) -> bool {
        self.layout.is_some()
            || self.calibration.is_some()
            || self.no_calibration
            || self.strict_calibration
            || self.auto_tau.is_some()
            || self.auto_band.is_some()
    }
    /// Explicit shared CLI values override per-model YAML; one calibration source replaces the other.
    pub(crate) fn apply_cli(&mut self, cli: &Self) {
        if cli.layout.is_some() {
            self.layout.clone_from(&cli.layout);
        }
        if cli.calibration.is_some() || cli.no_calibration {
            self.calibration.clone_from(&cli.calibration);
            self.no_calibration = cli.no_calibration;
        }
        self.strict_calibration |= cli.strict_calibration;
        if cli.auto_tau.is_some() {
            self.auto_tau = cli.auto_tau;
        }
        if cli.auto_band.is_some() {
            self.auto_band.clone_from(&cli.auto_band);
        }
    }

    pub(crate) fn auto_orders(&self) -> Result<AutoOrderSettings> {
        let band = match self.auto_band.as_deref() {
            None => [0.2, 0.8],
            Some([lo, hi]) => [*lo, *hi],
            Some(_) => anyhow::bail!("auto_band requires exactly two numbers"),
        };
        Ok(AutoOrderSettings::new(self.auto_tau.unwrap_or(0.6), band)?)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(!(self.no_calibration && self.calibration.is_some()), "calibration conflicts with no_calibration");
        if let Some(layout) = &self.layout {
            PromptLayout::parse(layout)?;
        }
        self.auto_orders()?;
        Ok(())
    }

    /// Identity follows upstream: HF snapshot repo/commit, otherwise the actual configured source and no revision.
    /// Never infer identity from the routed name or basal.json. Validate before weights or GEMM allocation.
    pub(crate) fn load_manifest(&self, dir: &Path, orders: Option<&str>, source: &str) -> Result<ModelManifest> {
        self.validate()?;
        let snapshot_repo = dir
            .parent()
            .filter(|p| p.file_name().is_some_and(|n| n == "snapshots"))
            .and_then(Path::parent)
            .and_then(Path::file_name)
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_prefix("models--"))
            .map(|n| n.replace("--", "/"));
        let revision = snapshot_repo.as_ref().and_then(|_| dir.file_name()).and_then(|n| n.to_str());
        let manifest = ModelManifest::load_with_options(
            dir,
            ModelLoadOptions {
                orders,
                layout: self.layout.as_deref().map(PromptLayout::parse).transpose()?,
                calibration: if self.no_calibration {
                    CalibrationSource::Off
                } else if let Some(path) = &self.calibration {
                    CalibrationSource::File(path)
                } else {
                    CalibrationSource::Automatic
                },
                strict_calibration: self.strict_calibration,
                served_identity: Some(ServedIdentity { model: snapshot_repo.as_deref().unwrap_or(source), revision }),
            },
        )
        .context("loading model prompt and calibration settings")?;
        if let Some(warning) = calibration_warning(&manifest) {
            crate::warn!("{}: {warning}", manifest.name);
        }
        Ok(manifest)
    }
}

/// Metadata/log warning without local source identities or the calibration's full path.
pub(crate) fn calibration_warning(manifest: &ModelManifest) -> Option<String> {
    let mut warnings = Vec::new();
    if let Some(warning) = &manifest.calibration_unvalidated {
        warnings.push(warning.as_str());
    }
    if manifest.calibration_provenance.identity_warning.is_some() {
        warnings.push("explicit calibration was fitted on another model or revision; calibration retained");
    }
    (!warnings.is_empty()).then(|| warnings.join("; "))
}
