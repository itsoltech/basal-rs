//! `basal database`: administration of the configured database without loading a model or a GPU.
//!
//! - `migrate`: create or migrate the schema with the migration account (SQLite: create the file), whatever
//!   `auto_migrate` says. It installs no records of a consumer: the Batch quota policy and model catalog stay with
//!   `basal batches install`.
//!
//! The configuration is the root `database:` section of the serve configuration (`--config FILE`, `BASAL_CONFIG` or
//! `basal-serve.yml`); `models` and the consumers' sections may be absent or disabled.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use super::{DatabaseConfig, DatabaseSection};

#[derive(clap::Args)]
pub(crate) struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Create or migrate the database schema with the migration account (no model, no GPU, no consumer records)
    Migrate {
        /// Configuration with a `database:` section (default: BASAL_CONFIG, else basal-serve.yml here)
        #[arg(long)]
        config: Option<PathBuf>,
    },
}

pub(crate) fn run(args: Args) -> Result<()> {
    match args.command {
        Command::Migrate { config } => {
            let file = config
                .or_else(crate::paths::config_file)
                .context("no configuration: pass --config FILE with a `database:` section")?;
            let config = crate::config::ServeConfig::load_admin(&file)?;
            config.batches.check_moved_settings()?;
            let settings = match DatabaseConfig::section(config.database.as_ref())? {
                DatabaseSection::Enabled(settings) => settings,
                DatabaseSection::Disabled => bail!("database.enabled is false in {}", crate::term::P(&file)),
                DatabaseSection::Absent => bail!(
                    "no `database:` section in {} (the local Batch default without one: `basal batches migrate`)",
                    crate::term::P(&file)
                ),
            };
            // Both rustls providers are in the dependency graph; the drivers need one process default.
            let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
            let runtime =
                tokio::runtime::Builder::new_multi_thread().enable_all().build().context("starting the runtime")?;
            let state = runtime.block_on(super::migrate(&settings))?;
            crate::done!("database schema ({}): {}", settings.engine_label(), super::describe_schema(&state));
            Ok(())
        }
    }
}
