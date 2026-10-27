//! `basal batches`: administration of durable batch storage without loading a model or a GPU.
//!
//! - `install`: with the migration account, install the quota policy of the configuration when it differs from the
//!   installed one, and a model catalog revision from `--catalog FILE`. Needs a current schema (no DDL). Run it with
//!   batch admission paused when the policy changes.
//! - `migrate`: compatibility shortcut of `basal database migrate` (schema only) followed by `install`; it also
//!   works for the local default database of `basal serve --batches`, which has no `database:` section.
//! - `doctor`: connection with the runtime account (TLS as configured), schema, installed policy and catalog,
//!   bounded counters, quota drift and artifact storage, without printing connection strings or client data.
//!
//! The configuration is the `batches:` and root `database:` sections of the serve configuration (`--config FILE`,
//! `BASAL_CONFIG` or `basal-serve.yml`; `models` may be absent); without one, the local profile under
//! `--batch-data-dir` or the user data directory, with its SQLite file `batches.sqlite3` there.
//!
//! Catalog file (`--catalog`): `{"version": 1, "default_model": "<name>", "models": [{"name": "<name>",
//! "fingerprint": "<fp1-…>", "planner_version": "basal-planner.2"}]}`. `basal serve` logs the fingerprint of every
//! served model and `/health/batches` lists them; a new revision applies to batches created after it, and queued
//! batches keep the revision they pinned.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::json;

use crate::database::{Account, DatabaseConfig, DatabaseSettings, Target};
use crate::serve::batches::artifacts::ArtifactStore;
use crate::serve::batches::config::{BatchSettings, ValidateContext};
use crate::serve::batches::store::{
    self, CatalogEntry, CatalogSpec, Fingerprint, InstallOutcome, ModelName, PlannerVersion, SchemaState, Store,
};

#[derive(clap::Args)]
pub(crate) struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// `basal database migrate` (schema) and then `install`; also for the local default database
    Migrate {
        #[command(flatten)]
        source: Source,
        /// Model catalog JSON to install as a new revision (see `basal batches --help`)
        #[arg(long)]
        catalog: Option<PathBuf>,
    },
    /// Install the quota policy and (with --catalog) a catalog revision; the schema must be current
    Install {
        #[command(flatten)]
        source: Source,
        /// Model catalog JSON to install as a new revision (see `basal batches --help`)
        #[arg(long)]
        catalog: Option<PathBuf>,
    },
    /// Check batch storage, schema, policy, catalog and artifacts (no model, no GPU)
    Doctor {
        #[command(flatten)]
        source: Source,
        /// Print one JSON object
        #[arg(long)]
        json: bool,
    },
}

#[derive(clap::Args)]
struct Source {
    /// Serve configuration with `batches:` and `database:` sections (default: BASAL_CONFIG, else basal-serve.yml
    /// here)
    #[arg(long)]
    config: Option<PathBuf>,
    /// Local batch data directory (local profile without a configuration file)
    #[arg(long)]
    batch_data_dir: Option<PathBuf>,
}

pub(crate) fn run(args: Args) -> Result<()> {
    match args.command {
        Command::Migrate { source, catalog } => {
            let (settings, database) = settings(&source)?;
            runtime()?.block_on(async {
                let state = crate::database::migrate(&database).await?;
                crate::done!("schema: {}", crate::database::describe_schema(&state));
                install(&settings, &database, catalog.as_deref()).await
            })
        }
        Command::Install { source, catalog } => {
            let (settings, database) = settings(&source)?;
            runtime()?.block_on(install(&settings, &database, catalog.as_deref()))
        }
        Command::Doctor { source, json } => {
            let (settings, database) = settings(&source)?;
            runtime()?.block_on(doctor(&settings, &database, json))
        }
    }
}

fn runtime() -> Result<tokio::runtime::Runtime> {
    // Both rustls providers are in the dependency graph; the drivers need one process default.
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    tokio::runtime::Builder::new_multi_thread().enable_all().build().context("starting the runtime")
}

/// The Batch settings (enabled whatever the file says) and the database they use.
fn settings(source: &Source) -> Result<(BatchSettings, DatabaseSettings)> {
    let file = source.config.clone().or_else(crate::paths::config_file);
    let mut config = match &file {
        Some(path) => crate::config::ServeConfig::load_admin(path)?,
        None => crate::config::ServeConfig::for_models(Vec::new()),
    };
    let database = DatabaseConfig::section(config.database.as_ref())?;
    config.batches.apply_cli(true, source.batch_data_dir.clone());
    let ctx = ValidateContext { default_data_dir: crate::paths::data_dir() };
    let settings = config.batches.validate(&ctx)?.context("the batch configuration is disabled")?;
    let database = settings.database(&database)?;
    Ok((settings, database))
}

/// Open the store with `account` without running DDL; a missing SQLite file is reported, not created.
async fn open(settings: &BatchSettings, database: &DatabaseSettings, account: Account) -> Result<Store> {
    let options = crate::database::options(database, account).await?;
    if let Target::Sqlite { path } = &database.target {
        if !path.exists() {
            bail!(
                "no batch database at {}; run `basal database migrate` (without a `database:` section: \
                 `basal batches migrate`) first",
                crate::term::P(path)
            );
        }
    }
    Ok(crate::serve::batches::connect_store(&options, settings.store_settings()).await?)
}

fn schema(state: &SchemaState) -> String {
    crate::database::describe_schema(state)
}

/// Install policy and catalog with the migration account; the schema must be current.
async fn install(settings: &BatchSettings, database: &DatabaseSettings, catalog: Option<&Path>) -> Result<()> {
    let store = open(settings, database, Account::Migration).await?;
    let result = install_records(settings, &store, catalog).await;
    store.close().await;
    result
}

async fn install_records(settings: &BatchSettings, store: &Store, catalog: Option<&Path>) -> Result<()> {
    let spec = settings.quota.to_store_spec(&settings.limits)?;
    let digest = store::policy_digest(&spec);
    match store.active_quota_policy().await? {
        Some(active) if active.spec_sha256 == digest => {
            crate::done!("quota policy: revision {} is current", active.revision.0)
        }
        previous => {
            if previous.is_some() {
                crate::warn!("the installed quota policy differs; installing a new revision (pause batch admission)");
            }
            match store.install_quota_policy(&spec).await? {
                InstallOutcome::Installed(r) => crate::done!("quota policy: installed revision {}", r.0),
                InstallOutcome::Unchanged(r) => crate::done!("quota policy: revision {} is current", r.0),
            }
        }
    }
    match catalog {
        Some(path) => match store.install_catalog(&read_catalog(path)?).await? {
            InstallOutcome::Installed(r) => crate::done!("catalog: installed revision {}", r.0),
            InstallOutcome::Unchanged(r) => crate::done!("catalog: revision {} is current", r.0),
        },
        None => match store.latest_catalog().await? {
            Some(c) => crate::note!("catalog: revision {} ({} models)", c.revision.0, c.spec.entries.len()),
            None => crate::note!(
                "catalog: none installed (the local profile installs one at the first `basal serve --batches`; \
                 production: --catalog FILE)"
            ),
        },
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogFile {
    version: u32,
    default_model: String,
    models: Vec<CatalogModel>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogModel {
    name: String,
    fingerprint: String,
    planner_version: String,
}

fn read_catalog(path: &Path) -> Result<CatalogSpec> {
    let text = std::fs::read(path).with_context(|| format!("reading {}", crate::term::P(&path)))?;
    let file: CatalogFile =
        serde_json::from_slice(&text).with_context(|| format!("parsing {}", crate::term::P(&path)))?;
    if file.version != 1 {
        bail!("catalog version {} is not supported (1)", file.version);
    }
    let entries = file
        .models
        .iter()
        .map(|m| {
            Ok(CatalogEntry {
                model_name: ModelName::parse(&m.name)?,
                fingerprint: Fingerprint::parse(&m.fingerprint)?,
                planner_version: PlannerVersion::parse(&m.planner_version)?,
            })
        })
        .collect::<Result<Vec<_>, store::InvalidValue>>()?;
    Ok(CatalogSpec { default_model: ModelName::parse(&file.default_model)?, entries })
}

async fn doctor(settings: &BatchSettings, database: &DatabaseSettings, as_json: bool) -> Result<()> {
    let mut problems: Vec<String> = Vec::new();
    let mut report = json!({"profile": format!("{:?}", settings.profile).to_lowercase(),
                            "storage": {"engine": database.engine_label()}});
    match open(settings, database, Account::Runtime).await {
        Ok(store) => {
            match store.doctor().await {
                Ok(d) => {
                    report["storage"]["backend"] = json!(d.backend.as_str());
                    report["storage"]["server_version"] = json!(d.server_version);
                    report["storage"]["schema"] = json!(schema(&d.schema));
                    report["storage"]["quota_policy_revision"] = json!(d.quota_policy.map(|r| r.0));
                    report["storage"]["catalog_revision"] = json!(d.catalog.map(|r| r.0));
                    report["storage"]["batches_by_status"] =
                        d.batches_by_status.iter().map(|(s, n)| json!({"status": s.as_db(), "batches": n})).collect();
                    report["storage"]["expired_claims"] = json!(d.expired_claims);
                    report["storage"]["unaccounted_settlements"] = json!(d.unaccounted_settlements);
                    report["storage"]["pending_cleanup"] = json!(d.pending_cleanup);
                    report["storage"]["open_sessions"] = json!(d.sessions_open);
                    if !matches!(d.schema, SchemaState::Current { .. }) {
                        problems.push(format!("schema is {}", schema(&d.schema)));
                    }
                }
                Err(e) => problems.push(format!("storage report: {e}")),
            }
            let caps = store.capabilities();
            report["storage"]["capabilities"] = json!({
                "backend": caps.backend.as_str(),
                "transactional_ddl": caps.transactional_ddl,
                "row_locks": caps.row_locks,
                "skip_locked": caps.skip_locked,
                "server_clock": caps.server_clock,
            });
            let s = store.settings();
            report["storage"]["settings_s"] = json!({
                "discovery_lease": s.discovery_lease.as_secs(), "validation_lease": s.validation_lease.as_secs(),
                "execution_lease": s.execution_lease.as_secs(), "finalization_lease": s.finalization_lease.as_secs(),
                "upload_lease": s.upload_lease.as_secs(), "io_guard": s.io_guard.as_secs(),
                "file_retention": s.file_retention.as_secs(), "metadata_retention": s.metadata_retention.as_secs(),
                "download_grace": s.download_grace.as_secs(),
            });
            let spec = settings.quota.to_store_spec(&settings.limits)?;
            // Aggregate usage of the policy's owners (at most 100 read); owner ids are not printed.
            let (mut used, mut reserved, mut near_limit, mut read) = (0u64, 0u64, 0u64, 0u64);
            for (owner, limits) in spec.owners.iter().take(100) {
                match store.quota_usage(owner).await {
                    Ok(u) => {
                        read += 1;
                        used = used.saturating_add(u.used_bytes);
                        reserved = reserved.saturating_add(u.reserved_bytes);
                        if u.used_bytes.saturating_add(u.reserved_bytes) >= limits.max_bytes / 10 * 9 {
                            near_limit += 1;
                        }
                    }
                    Err(e) => problems.push(format!("quota usage: {e}")),
                }
            }
            report["storage"]["quota"] = json!({"owners_read": read, "owners_in_policy": spec.owners.len(),
                "used_bytes": used, "reserved_bytes": reserved, "owners_at_90_percent": near_limit});
            match store.active_quota_policy().await {
                Ok(Some(p)) if p.spec_sha256 == store::policy_digest(&spec) => {
                    report["storage"]["policy_matches_config"] = json!(true)
                }
                Ok(Some(_)) => {
                    report["storage"]["policy_matches_config"] = json!(false);
                    problems.push("the installed quota policy differs from the configuration".into());
                }
                Ok(None) => problems.push("no quota policy is installed".into()),
                Err(e) => problems.push(format!("quota policy: {e}")),
            }
            match store.quota_drift(100).await {
                Ok(drift) => {
                    report["storage"]["quota_drift_owners"] = json!(drift.len());
                    if !drift.is_empty() {
                        problems.push(format!("quota totals of {} owner(s) differ from their rows", drift.len()));
                    }
                }
                Err(e) => problems.push(format!("quota drift: {e}")),
            }
            store.close().await;
        }
        Err(e) => problems.push(format!("storage: {e:#}")),
    }
    let buffer = usize::try_from(settings.host.upload_buffer_bytes).unwrap_or(1 << 20);
    match ArtifactStore::open(&settings.artifacts, buffer).await {
        Ok(artifacts) => {
            report["artifacts"] = json!({"backend": artifacts.backend_label()});
            match artifacts.probe().await {
                Ok(()) => report["artifacts"]["probe"] = json!("ok"),
                Err(e) => problems.push(format!("artifacts probe: {e}")),
            }
        }
        Err(e) => problems.push(format!("artifacts: {e}")),
    }
    report["problems"] = json!(problems);
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !as_json {
        for p in &problems {
            crate::warn!("{p}");
        }
    }
    if !problems.is_empty() {
        bail!("batch doctor found {} problem(s)", problems.len());
    }
    Ok(())
}
