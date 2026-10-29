//! `basal keys`: API keys of the server (`auth:`, `serve::auth`) in the configured database, without a model or a GPU.
//!
//! - `create`: a new key of an owner with scopes and an optional lifetime. The token goes to stdout once (the
//!   database keeps only its SHA-256); everything else goes to stderr.
//! - `list`: keys without their tokens (`--all` also revoked ones, `--json` one JSON array).
//! - `revoke`: the key stops working on every server within `serve::auth::CACHE_TTL`.
//! - `rotate`: a new key with the owner, name and scopes of an old one; the old one stays valid until it is revoked
//!   or, with `--expire-old-in`, until that time.
//!
//! The database is the root `database:` section of the serve configuration (`--config FILE`, `BASAL_CONFIG` or
//! `basal-serve.yml`), else the local SQLite default of `basal serve --batches`. Records are written with the runtime
//! account; with `auto_migrate` (the SQLite default) a missing schema is created first, otherwise run
//! `basal database migrate`.

use std::io::Write as _;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde_json::json;

use crate::batches::Source;
use crate::database::{Database, DatabaseConfig, DatabaseSection, DatabaseSettings};
use crate::serve::auth::generate_token;
use crate::serve::batches::store::{
    check_api_key_name, ApiKeyId, ApiKeyRecord, ApiScopes, NewApiKey, OwnerId, Store, StoreError, UnixMillis,
};

/// Most keys `list` prints.
const MAX_LIST: u32 = 10_000;

#[derive(clap::Args)]
pub(crate) struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Create a key and print its token once (stdout)
    Create {
        #[command(flatten)]
        source: Source,
        /// Batch storage namespace of the key's requests (an owner of batches.quota.owners)
        #[arg(long, value_parser = parse_owner)]
        owner: OwnerId,
        /// Label for operators, e.g. the client or team (1-128 bytes, no control characters)
        #[arg(long, value_parser = parse_name)]
        name: String,
        /// Comma list of inference, batches, mcp, metrics
        #[arg(long, default_value = "inference,batches", value_parser = parse_scopes)]
        scopes: ApiScopes,
        /// Lifetime, e.g. 90d, 12h, 30m (default: no expiry)
        #[arg(long, value_parser = parse_duration)]
        expires_in: Option<Duration>,
        /// Print one JSON object (with the token) instead of the token alone
        #[arg(long)]
        json: bool,
    },
    /// List keys (never their tokens)
    List {
        #[command(flatten)]
        source: Source,
        /// Only keys of this owner
        #[arg(long, value_parser = parse_owner)]
        owner: Option<OwnerId>,
        /// Include revoked keys
        #[arg(long)]
        all: bool,
        /// Print one JSON array
        #[arg(long)]
        json: bool,
    },
    /// Revoke a key; servers refuse it within 30 s
    Revoke {
        #[command(flatten)]
        source: Source,
        /// Key id (the 16 hex characters after `bsk_`)
        #[arg(value_parser = parse_key_id)]
        key_id: ApiKeyId,
    },
    /// Create a successor of a key (same owner, name and scopes) and print its token once
    Rotate {
        #[command(flatten)]
        source: Source,
        /// Key id of the old key
        #[arg(value_parser = parse_key_id)]
        key_id: ApiKeyId,
        /// Let the old key expire after this time, e.g. 24h (default: keep it until `revoke`)
        #[arg(long, value_parser = parse_duration)]
        expire_old_in: Option<Duration>,
        /// Lifetime of the new key (default: no expiry)
        #[arg(long, value_parser = parse_duration)]
        expires_in: Option<Duration>,
        /// Print one JSON object (with the token) instead of the token alone
        #[arg(long)]
        json: bool,
    },
}

pub(crate) fn run(args: Args) -> Result<()> {
    let source = match &args.command {
        Command::Create { source, .. }
        | Command::List { source, .. }
        | Command::Revoke { source, .. }
        | Command::Rotate { source, .. } => source,
    };
    let database = database(source)?;
    crate::batches::runtime()?.block_on(async {
        let store = crate::serve::auth::connect(&Database::new(database)).await?;
        let result = command(&store, args.command).await;
        store.close().await;
        result
    })
}

async fn command(store: &Store, command: Command) -> Result<()> {
    match command {
        Command::Create { owner, name, scopes, expires_in, json, .. } => {
            check_quota_owner(store, &owner).await?;
            create(store, owner, name, scopes, expires_in, json).await
        }
        Command::List { owner, all, json, .. } => {
            let keys = store.list_api_keys(owner.as_ref(), all, MAX_LIST).await?;
            if json {
                println!("{}", serde_json::Value::Array(keys.iter().map(describe).collect()));
            } else {
                print_table(&keys);
            }
            Ok(())
        }
        Command::Revoke { key_id: id, .. } => {
            let key = store.revoke_api_key(&id).await.map_err(|e| not_found(e, &id))?;
            crate::done!("revoked key {} ({}, owner {}); servers refuse it within 30 s", id, key.name, key.owner);
            Ok(())
        }
        Command::Rotate { key_id: id, expire_old_in, expires_in, json, .. } => {
            let old = store.lookup_api_key(&id).await?.ok_or_else(|| not_found(StoreError::NotFound, &id))?.record;
            if old.revoked_at.is_some() {
                bail!("key {id} is revoked; create a new key instead");
            }
            create(store, old.owner, old.name, old.scopes, expires_in, json).await?;
            match expire_old_in {
                Some(within) => {
                    let old = store.expire_api_key(&id, within).await?;
                    crate::note!("old key {id} expires at {}", old.expires_at.map_or_else(|| "-".into(), utc));
                }
                None => crate::note!("old key {id} stays valid; revoke it with `basal keys revoke {id}`"),
            }
            Ok(())
        }
    }
}

async fn create(
    store: &Store,
    owner: OwnerId,
    name: String,
    scopes: ApiScopes,
    expires_in: Option<Duration>,
    json: bool,
) -> Result<()> {
    let token = generate_token().context("reading system randomness")?;
    let key = NewApiKey { id: token.id, owner, name, token_sha256: token.sha256, scopes, expires_in };
    let record = store.create_api_key(&key).await?;
    crate::done!(
        "created key {} for owner {} (scopes {}, expires {}); the token is shown only now",
        record.id,
        record.owner,
        record.scopes.to_text(),
        record.expires_at.map_or_else(|| "never".into(), utc)
    );
    let line = if json {
        let mut value = describe(&record);
        value["token"] = json!(token.token.expose());
        value.to_string()
    } else {
        token.token.expose().to_owned()
    };
    // A failed write (closed pipe) is an error, not a panic; the key then has to be revoked.
    writeln!(std::io::stdout(), "{line}")
        .with_context(|| format!("writing the token of key {}; revoke it with `basal keys revoke`", record.id))
}

/// Warn when an installed quota policy has no limits for `owner`: its Batch/Files requests would be refused.
async fn check_quota_owner(store: &Store, owner: &OwnerId) -> Result<()> {
    if let Some(policy) = store.active_quota_policy().await? {
        if !policy.spec.owners.iter().any(|(o, _)| o == owner) {
            crate::warn!(
                "owner {owner} has no limits in the installed quota policy: its Batch/Files requests are refused \
                 until batches.quota.owners lists it (`basal batches install`)"
            );
        }
    }
    Ok(())
}

/// The database of the keys: the root `database:` section, else the local default of `basal serve --batches`.
fn database(source: &Source) -> Result<DatabaseSettings> {
    if let Some(path) = source.config.clone().or_else(crate::paths::config_file) {
        let config = crate::config::ServeConfig::load_admin(&path)?;
        match DatabaseConfig::section(config.database.as_ref())? {
            DatabaseSection::Enabled(settings) => return Ok(settings),
            DatabaseSection::Disabled => bail!("database.enabled is false in {}", crate::term::P(&path)),
            DatabaseSection::Absent => {}
        }
    }
    Ok(crate::batches::settings(source)?.1)
}

fn not_found(error: StoreError, id: &ApiKeyId) -> anyhow::Error {
    match error {
        StoreError::NotFound => anyhow::anyhow!("no key {id}"),
        other => other.into(),
    }
}

/// Status for display, by this host's clock (servers decide by the database clock).
fn status(key: &ApiKeyRecord) -> &'static str {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));
    if key.revoked_at.is_some() {
        "revoked"
    } else if key.active_at(UnixMillis(now)) {
        "active"
    } else {
        "expired"
    }
}

fn describe(key: &ApiKeyRecord) -> serde_json::Value {
    let time = |t: Option<UnixMillis>| t.map(utc);
    json!({
        "key_id": key.id.as_str(),
        "owner": key.owner.as_str(),
        "name": key.name,
        "scopes": key.scopes.to_text(),
        "created_at": utc(key.created_at),
        "expires_at": time(key.expires_at),
        "revoked_at": time(key.revoked_at),
        "last_used_at": time(key.last_used_at),
        "status": status(key),
    })
}

fn print_table(keys: &[ApiKeyRecord]) {
    let dash = |t: Option<UnixMillis>| t.map_or_else(|| "-".to_string(), utc);
    println!(
        "{:<16}  {:<20}  {:<24}  {:<28}  {:<20}  {:<20}  {:<20}  STATUS",
        "KEY_ID", "OWNER", "NAME", "SCOPES", "CREATED", "EXPIRES", "LAST_USED"
    );
    for key in keys {
        println!(
            "{:<16}  {:<20}  {:<24}  {:<28}  {:<20}  {:<20}  {:<20}  {}",
            key.id.as_str(),
            key.owner.as_str(),
            key.name.escape_debug().to_string(),
            key.scopes.to_text(),
            utc(key.created_at),
            dash(key.expires_at),
            dash(key.last_used_at),
            status(key)
        );
    }
}

/// `2026-10-29T12:00:00Z` of a database time (proleptic Gregorian, UTC).
fn utc(t: UnixMillis) -> String {
    let secs = t.0.div_euclid(1000);
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil date of days since 1970-01-01 (H. Hinnant, `civil_from_days`).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

fn parse_owner(text: &str) -> Result<OwnerId, String> {
    OwnerId::parse(text).map_err(|e| e.to_string())
}

fn parse_name(text: &str) -> Result<String, String> {
    check_api_key_name(text).map(|()| text.to_owned()).map_err(|e| e.to_string())
}

fn parse_scopes(text: &str) -> Result<ApiScopes, String> {
    ApiScopes::parse(text).map_err(|e| e.to_string())
}

fn parse_key_id(text: &str) -> Result<ApiKeyId, String> {
    ApiKeyId::parse(text).map_err(|e| e.to_string())
}

/// `90d`, `12h`, `30m` or `45s`: a positive whole number of days, hours, minutes or seconds (at most 100 years).
fn parse_duration(text: &str) -> Result<Duration, String> {
    let (number, unit) = text.split_at_checked(text.len().saturating_sub(1)).unwrap_or(("", ""));
    let scale: u64 = match unit {
        "d" => 86_400,
        "h" => 3600,
        "m" => 60,
        "s" => 1,
        _ => return Err("use a number with d, h, m or s, e.g. 90d".into()),
    };
    let secs = number
        .parse::<u64>()
        .ok()
        .and_then(|n| n.checked_mul(scale))
        .filter(|s| (1..=100 * 365 * 86_400).contains(s))
        .ok_or("use a positive whole number of at most 100 years, e.g. 90d")?;
    Ok(Duration::from_secs(secs))
}
