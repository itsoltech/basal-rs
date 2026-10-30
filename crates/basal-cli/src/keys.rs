//! `basal users` and `basal keys`: users and API keys of the server (`auth:`, `serve::auth`) in the configured
//! database, without a model or a GPU.
//!
//! A user's id is the Batch storage namespace (owner) of all its keys: every key of a user sees the same files and
//! batches; the user's Batch limits stay in `batches.quota.owners`.
//!
//! - `users create|list|disable|enable`: a disabled user's keys are refused by every server within
//!   `serve::auth::CACHE_TTL` (its UI sessions at once); its data stays and expires by the Batch retention.
//!   `create --admin` makes an admin of the browser UI (`serve::ui`).
//! - `users password`: set (from stdin) or clear the user's UI password; its UI sessions end.
//! - `users delete`: delete a user disabled at least `CACHE_TTL` ago without Batch data, with its keys and sessions.
//! - `keys create`: a new key of an enabled user with scopes and an optional lifetime. The token goes to stdout once
//!   (the database keeps only its SHA-256); everything else goes to stderr.
//! - `keys list`: keys without their tokens (`--all` also revoked ones, `--json` one JSON array).
//! - `keys revoke`: the key stops working on every server within `serve::auth::CACHE_TTL`.
//! - `keys rotate`: a new key with the user, name and scopes of an old one; the old one stays valid until it is
//!   revoked or, with `--expire-old-in`, until that time.
//!
//! The database is the root `database:` section of the serve configuration (`--config FILE`, `BASAL_CONFIG` or
//! `basal-serve.yml`), else the local server database (`DatabaseSettings::local_default`). Records are written with the runtime
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
    check_api_name, ApiKeyId, ApiKeyRecord, ApiRole, ApiScopes, ApiUserRecord, NewApiKey, OwnerId, PasswordHash, Store,
    StoreError, UnixMillis,
};

/// Most users or keys `list` prints.
const MAX_LIST: u32 = 10_000;

#[derive(clap::Args)]
pub(crate) struct UsersArgs {
    #[command(subcommand)]
    command: UsersCommand,
}

#[derive(clap::Subcommand)]
enum UsersCommand {
    /// Create a user (its id is the Batch storage namespace of all its keys)
    Create {
        #[command(flatten)]
        source: Source,
        /// User id: 1-128 of A-Z, a-z, 0-9, `._:@-`; also its owner in batches.quota.owners
        #[arg(value_parser = parse_user)]
        user: OwnerId,
        /// Label for operators, e.g. the client or team (1-128 bytes, no control characters)
        #[arg(long, value_parser = parse_name)]
        name: String,
        /// Admin of the browser UI (manages every user and key there)
        #[arg(long)]
        admin: bool,
        /// Read the UI password from the first line of stdin (at least 12 characters)
        #[arg(long)]
        password_stdin: bool,
    },
    /// Set the UI password of a user from the first line of stdin, or clear it; its UI sessions end
    Password {
        #[command(flatten)]
        source: Source,
        #[arg(value_parser = parse_user)]
        user: OwnerId,
        /// Remove the password: the user can no longer log in to the UI (its keys still work)
        #[arg(long)]
        clear: bool,
    },
    /// Delete a user disabled at least 30 s ago without Batch data, with its keys and UI sessions
    Delete {
        #[command(flatten)]
        source: Source,
        #[arg(value_parser = parse_user)]
        user: OwnerId,
    },
    /// List users
    List {
        #[command(flatten)]
        source: Source,
        /// Include disabled users
        #[arg(long)]
        all: bool,
        /// Print one JSON array
        #[arg(long)]
        json: bool,
    },
    /// Disable a user: servers refuse all its keys within 30 s; its data stays
    Disable {
        #[command(flatten)]
        source: Source,
        #[arg(value_parser = parse_user)]
        user: OwnerId,
    },
    /// Enable a disabled user again (its unrevoked, unexpired keys work again)
    Enable {
        #[command(flatten)]
        source: Source,
        #[arg(value_parser = parse_user)]
        user: OwnerId,
    },
}

#[derive(clap::Args)]
pub(crate) struct KeysArgs {
    #[command(subcommand)]
    command: KeysCommand,
}

#[derive(clap::Subcommand)]
enum KeysCommand {
    /// Create a key of a user and print its token once (stdout)
    Create {
        #[command(flatten)]
        source: Source,
        /// User of the key (`basal users create`)
        #[arg(long, value_parser = parse_user)]
        user: OwnerId,
        /// Label for operators, e.g. the application or host (1-128 bytes, no control characters)
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
        /// Only keys of this user
        #[arg(long, value_parser = parse_user)]
        user: Option<OwnerId>,
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
    /// Create a successor of a key (same user, name and scopes) and print its token once
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

pub(crate) fn run_users(args: UsersArgs) -> Result<()> {
    let source = match &args.command {
        UsersCommand::Create { source, .. }
        | UsersCommand::Password { source, .. }
        | UsersCommand::Delete { source, .. }
        | UsersCommand::List { source, .. }
        | UsersCommand::Disable { source, .. }
        | UsersCommand::Enable { source, .. } => source,
    };
    let database = database(source)?;
    // Read and hash before connecting, so a refused password opens no database.
    let password = match &args.command {
        UsersCommand::Create { password_stdin: true, .. } | UsersCommand::Password { clear: false, .. } => {
            Some(read_password()?)
        }
        _ => None,
    };
    with_store(database, async |store: &Store| users(store, args.command, password.as_ref()).await)
}

pub(crate) fn run_keys(args: KeysArgs) -> Result<()> {
    let source = match &args.command {
        KeysCommand::Create { source, .. }
        | KeysCommand::List { source, .. }
        | KeysCommand::Revoke { source, .. }
        | KeysCommand::Rotate { source, .. } => source,
    };
    let database = database(source)?;
    with_store(database, async |store: &Store| keys(store, args.command).await)
}

/// Run `f` with the store of `database` and close the store afterwards.
fn with_store(database: DatabaseSettings, f: impl AsyncFnOnce(&Store) -> Result<()>) -> Result<()> {
    crate::batches::runtime()?.block_on(async {
        let store = crate::serve::auth::connect(&Database::new(database)).await?;
        let result = f(&store).await;
        store.close().await;
        result
    })
}

/// Hash of the password on the first line of stdin (without its line end); refused when shorter than 12 characters.
fn read_password() -> Result<PasswordHash> {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).context("reading the password from stdin")?;
    // Zeroed when dropped.
    let line = crate::config_support::SecretString::new(line);
    let password = line.expose().trim_end_matches(['\r', '\n']);
    crate::serve::ui::session::check_password(password).map_err(|e| anyhow::anyhow!(e))?;
    crate::serve::ui::session::hash_password(password)
}

async fn users(store: &Store, command: UsersCommand, password: Option<&PasswordHash>) -> Result<()> {
    match command {
        UsersCommand::Create { user, name, admin, .. } => {
            let role = if admin { ApiRole::Admin } else { ApiRole::User };
            let record = store
                .create_api_user(&user, &name, role, password)
                .await?
                .with_context(|| format!("user {user} exists"))?;
            crate::done!(
                "created {} {} ({}){}; add keys with `basal keys create --user {}`",
                record.role.as_str(),
                user,
                record.name,
                if record.password.is_some() { " with a UI password" } else { "" },
                user
            );
            check_quota(store, &user).await;
            Ok(())
        }
        UsersCommand::Password { user, .. } => {
            store.set_api_user_password(&user, password, None).await.map_err(|e| user_not_found(e, &user))?;
            match password {
                Some(_) => crate::done!("set the UI password of user {user}; its UI sessions ended"),
                None => crate::done!("cleared the UI password of user {user}; its UI sessions ended"),
            }
            Ok(())
        }
        UsersCommand::Delete { user, .. } => {
            store.delete_api_user(&user, crate::serve::auth::CACHE_TTL).await.map_err(|e| match e {
                StoreError::UserNotDisabled => anyhow::anyhow!(
                    "user {user} is not disabled long enough: `basal users disable {user}`, then wait 30 s"
                ),
                StoreError::UserHasData => anyhow::anyhow!(
                    "user {user} still has Batch files, batches or upload sessions; they must expire first"
                ),
                e => user_not_found(e, &user),
            })?;
            crate::done!("deleted user {user} with its keys and UI sessions");
            Ok(())
        }
        UsersCommand::List { all, json, .. } => {
            let users = store.list_api_users(all, MAX_LIST).await?;
            if json {
                let rows = users.iter().map(|u| {
                    json!({
                        "user": u.id.as_str(),
                        "name": u.name,
                        "created_at": utc(u.created_at),
                        "disabled_at": u.disabled_at.map(utc),
                        "status": user_status(u),
                        "role": u.role.as_str(),
                        "ui_password": u.password.is_some(),
                    })
                });
                println!("{}", serde_json::Value::Array(rows.collect()));
            } else {
                println!(
                    "{:<24}  {:<32}  {:<20}  {:<5}  {:<11}  STATUS",
                    "USER", "NAME", "CREATED", "ROLE", "UI_PASSWORD"
                );
                for u in &users {
                    let name = u.name.escape_debug().to_string();
                    println!(
                        "{:<24}  {name:<32}  {:<20}  {:<5}  {:<11}  {}",
                        u.id.as_str(),
                        utc(u.created_at),
                        u.role.as_str(),
                        if u.password.is_some() { "yes" } else { "no" },
                        user_status(u)
                    );
                }
            }
            Ok(())
        }
        UsersCommand::Disable { user, .. } => {
            store.set_api_user_disabled(&user, true).await.map_err(|e| user_not_found(e, &user))?;
            crate::done!("disabled user {user}; servers refuse its keys within 30 s, its data stays");
            Ok(())
        }
        UsersCommand::Enable { user, .. } => {
            store.set_api_user_disabled(&user, false).await.map_err(|e| user_not_found(e, &user))?;
            crate::done!("enabled user {user}; its active keys work again within 30 s");
            Ok(())
        }
    }
}

async fn keys(store: &Store, command: KeysCommand) -> Result<()> {
    match command {
        KeysCommand::Create { user, name, scopes, expires_in, json, .. } => {
            create(store, user, name, scopes, expires_in, json).await
        }
        KeysCommand::List { user, all, json, .. } => {
            let keys = store.list_api_keys(user.as_ref(), all, MAX_LIST).await?;
            if json {
                println!("{}", serde_json::Value::Array(keys.iter().map(describe).collect()));
            } else {
                print_table(&keys);
            }
            Ok(())
        }
        KeysCommand::Revoke { key_id: id, .. } => {
            let key = store.revoke_api_key(&id).await.map_err(|e| key_not_found(e, &id))?;
            crate::done!("revoked key {} ({}, user {}); servers refuse it within 30 s", id, key.name, key.user);
            Ok(())
        }
        KeysCommand::Rotate { key_id: id, expire_old_in, expires_in, json, .. } => {
            let old = store.lookup_api_key(&id).await?.ok_or_else(|| key_not_found(StoreError::NotFound, &id))?.record;
            if old.revoked_at.is_some() {
                bail!("key {id} is revoked; create a new key instead");
            }
            create(store, old.user, old.name, old.scopes, expires_in, json).await?;
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
    user: OwnerId,
    name: String,
    scopes: ApiScopes,
    expires_in: Option<Duration>,
    json: bool,
) -> Result<()> {
    let token = generate_token().context("reading system randomness")?;
    let key = NewApiKey { id: token.id, user: user.clone(), name, token_sha256: token.sha256, scopes, expires_in };
    let record = store.create_api_key(&key).await.map_err(|e| match e {
        StoreError::NotFound => anyhow::anyhow!("no user {user}; create it with `basal users create {user}`"),
        StoreError::UserDisabled => anyhow::anyhow!("user {user} is disabled; `basal users enable {user}` first"),
        other => other.into(),
    })?;
    crate::done!(
        "created key {} for user {} (scopes {}, expires {}); the token is shown only now",
        record.id,
        record.user,
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

/// Warn when an installed quota policy has no limits for `user`: its Batch/Files requests would be refused. Runs
/// after the user is created, so a failed read is only a warning.
async fn check_quota(store: &Store, user: &OwnerId) {
    match store.active_quota_policy().await {
        Ok(Some(policy)) if !policy.spec.owners.iter().any(|(o, _)| o == user) => crate::warn!(
            "user {user} has no limits in the installed quota policy: its Batch/Files requests are refused \
             (403) until batches.quota.owners lists it (`basal batches install`)"
        ),
        Ok(_) => {}
        Err(e) => crate::warn!("user {user} is created, but reading the quota policy failed: {e}"),
    }
}

/// The database of users and keys: the root `database:` section, else the local server database.
fn database(source: &Source) -> Result<DatabaseSettings> {
    let file = source.config.clone().or_else(crate::paths::config_file);
    let mut config = match &file {
        Some(path) => crate::config::ServeConfig::load_admin(path)?,
        None => crate::config::ServeConfig::for_models(Vec::new()),
    };
    match DatabaseConfig::section(config.database.as_ref())? {
        DatabaseSection::Enabled(settings) => Ok(settings),
        DatabaseSection::Disabled => bail!("database.enabled is false in the configuration"),
        DatabaseSection::Absent => {
            config.batches.apply_cli(false, source.batch_data_dir.clone());
            Ok(DatabaseSettings::local_default(&crate::paths::data_dir(), &config.batches.local_data_dir()))
        }
    }
}

fn key_not_found(error: StoreError, id: &ApiKeyId) -> anyhow::Error {
    match error {
        StoreError::NotFound => anyhow::anyhow!("no key {id}"),
        other => other.into(),
    }
}

fn user_not_found(error: StoreError, user: &OwnerId) -> anyhow::Error {
    match error {
        StoreError::NotFound => anyhow::anyhow!("no user {user}"),
        other => other.into(),
    }
}

fn user_status(user: &ApiUserRecord) -> &'static str {
    if user.disabled_at.is_some() {
        "disabled"
    } else {
        "active"
    }
}

/// Status for display, by this host's clock (servers decide by the database clock). A disabled user is not shown
/// here; see `basal users list`.
pub(crate) fn status(key: &ApiKeyRecord) -> &'static str {
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
        "user": key.user.as_str(),
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
        "KEY_ID", "USER", "NAME", "SCOPES", "CREATED", "EXPIRES", "LAST_USED"
    );
    for key in keys {
        println!(
            "{:<16}  {:<20}  {:<24}  {:<28}  {:<20}  {:<20}  {:<20}  {}",
            key.id.as_str(),
            key.user.as_str(),
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

fn parse_user(text: &str) -> Result<OwnerId, String> {
    OwnerId::parse(text).map_err(|e| e.to_string())
}

fn parse_name(text: &str) -> Result<String, String> {
    check_api_name(text).map(|()| text.to_owned()).map_err(|e| e.to_string())
}

fn parse_scopes(text: &str) -> Result<ApiScopes, String> {
    ApiScopes::parse(text).map_err(|e| e.to_string())
}

fn parse_key_id(text: &str) -> Result<ApiKeyId, String> {
    ApiKeyId::parse(text).map_err(|e| e.to_string())
}

/// `90d`, `12h`, `30m` or `45s`: a positive whole number of days, hours, minutes or seconds (at most 100 years).
pub(crate) fn parse_duration(text: &str) -> Result<Duration, String> {
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
