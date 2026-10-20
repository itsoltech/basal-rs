//! Versioned MongoDB schema: collections and indexes from `migrations/mongodb/*.json`, recorded in `batch_schema`.
//!
//! MongoDB DDL is not transactional, so a migration writes its `batch_schema` entry as `applying` first, creates
//! collections and indexes (both idempotent for identical specifications), then marks it `applied`. An interrupted
//! migration is reported as `Damaged` and `migrate` resumes it when the embedded checksum still matches. Nothing is
//! dropped; a conflicting index specification fails the migration instead of being replaced. Runtime start only
//! reads the schema state.

use mongodb::bson::{doc, Bson, Document};
use mongodb::options::IndexOptions;
use mongodb::IndexModel;
use serde::Deserialize;

use super::super::error::StoreError;
use super::super::types::{hex, SchemaState};
use super::driver::{corrupt, int, majority_journal, redact, text, Core, SCHEMA};

struct Migration {
    version: i64,
    text: &'static str,
}

const MIGRATIONS: &[Migration] = &[
    Migration { version: 1, text: include_str!("../migrations/mongodb/0001_init.json") },
    Migration { version: 2, text: include_str!("../migrations/mongodb/0002_abandonment.json") },
];

#[derive(Deserialize)]
struct Spec {
    version: i64,
    id: String,
    collections: Vec<String>,
    indexes: Vec<IndexSpec>,
}

#[derive(Deserialize)]
struct IndexSpec {
    collection: String,
    name: String,
    keys: Vec<(String, i32)>,
    #[serde(default)]
    unique: bool,
    /// Index only documents where this field is a string (sparse claim references).
    #[serde(default)]
    partial_string: Option<String>,
}

impl Migration {
    fn checksum(&self) -> String {
        use sha2::{Digest, Sha256};
        hex(&Sha256::digest(self.text.as_bytes()))
    }

    fn spec(&self) -> Result<Spec, StoreError> {
        let spec: Spec = serde_json::from_str(self.text).map_err(|_| corrupt("embedded migration"))?;
        if spec.version != self.version {
            return Err(corrupt("embedded migration version"));
        }
        Ok(spec)
    }
}

fn expected() -> i64 {
    MIGRATIONS.last().map_or(0, |m| m.version)
}

/// Read-only schema check.
pub(super) async fn schema_state(core: &Core) -> Result<SchemaState, StoreError> {
    let rows = core.read_many(SCHEMA, doc! {}, doc! { "_id": 1_i32 }, 1000).await?;
    let expected = expected();
    let mut version = 0;
    for row in &rows {
        version = int(row, "_id")?;
        if version > expected {
            return Ok(SchemaState::TooNew { version, expected });
        }
        let known = MIGRATIONS.iter().find(|m| m.version == version);
        if text(row, "state")? != "applied"
            || known.is_none_or(|m| m.checksum() != text(row, "checksum").unwrap_or_default())
        {
            return Ok(SchemaState::Damaged { version });
        }
    }
    Ok(match version {
        0 => SchemaState::Empty,
        v if v == expected => SchemaState::Current { version },
        v => SchemaState::Outdated { version: v, expected },
    })
}

fn index_model(spec: &IndexSpec) -> IndexModel {
    let mut keys = Document::new();
    for (field, direction) in &spec.keys {
        keys.insert(field.clone(), *direction);
    }
    let mut options = IndexOptions::builder().name(spec.name.clone()).build();
    if spec.unique {
        options.unique = Some(true);
    }
    if let Some(field) = &spec.partial_string {
        options.partial_filter_expression = Some(doc! { field.clone(): { "$type": "string" } });
    }
    IndexModel::builder().keys(keys).options(options).build()
}

async fn apply(core: &Core, migration: &Migration) -> Result<(), StoreError> {
    let spec = migration.spec()?;
    let existing = core.db.list_collection_names().await.map_err(|e| redact(&e))?;
    for name in &spec.collections {
        if !existing.contains(name) {
            match core.db.create_collection(name.as_str()).await {
                Ok(()) => {}
                // 48 NamespaceExists: a concurrent or earlier interrupted run created it.
                Err(error) if matches!(redact(&error), StoreError::Database(ref db) if db.code.as_deref() == Some("48")) =>
                    {}
                Err(error) => return Err(redact(&error)),
            }
        }
    }
    for index in &spec.indexes {
        core.coll(&index.collection).create_index(index_model(index)).await.map_err(|e| redact(&e))?;
    }
    let _ = spec.id;
    Ok(())
}

/// Apply pending migrations (explicit `basal batches migrate`). Newer or damaged-with-other-checksum schemas are
/// refused; an interrupted migration with the same checksum is resumed.
pub(super) async fn migrate(core: &Core) -> Result<SchemaState, StoreError> {
    let schema = core.coll(SCHEMA);
    for migration in MIGRATIONS {
        let state = schema_state(core).await?;
        let applied = match state {
            SchemaState::Current { .. } => return Ok(state),
            SchemaState::Empty => 0,
            SchemaState::Outdated { version, .. } => version,
            SchemaState::TooNew { .. } => return Err(StoreError::Schema(state)),
            SchemaState::Damaged { version } => {
                let row = schema
                    .find_one(doc! { "_id": version })
                    .await
                    .map_err(|e| redact(&e))?
                    .ok_or(StoreError::Schema(state))?;
                let resumable = text(&row, "state")? == "applying"
                    && MIGRATIONS
                        .iter()
                        .any(|m| m.version == version && m.checksum() == text(&row, "checksum").unwrap_or_default());
                if !resumable {
                    return Err(StoreError::Schema(state));
                }
                version - 1
            }
        };
        if migration.version <= applied {
            continue;
        }
        let now = core.server_now().await?;
        let spec = migration.spec()?;
        schema
            .update_one(
                doc! { "_id": migration.version },
                doc! { "$setOnInsert": {
                    "migration_id": spec.id.as_str(),
                    "checksum": migration.checksum(),
                    "state": "applying",
                    "applied_at": now,
                } },
            )
            .upsert(true)
            .write_concern(majority_journal())
            .await
            .map_err(|e| redact(&e))?;
        apply(core, migration).await?;
        schema
            .update_one(
                doc! { "_id": migration.version, "checksum": migration.checksum() },
                doc! { "$set": { "state": "applied", "applied_at": now } },
            )
            .write_concern(majority_journal())
            .await
            .map_err(|e| redact(&e))?;
    }
    schema_state(core).await
}

/// `buildInfo.version` of the server.
pub(super) async fn server_version(core: &Core) -> Result<String, StoreError> {
    let reply = core.db.run_command(doc! { "buildInfo": 1_i32 }).await.map_err(|e| redact(&e))?;
    match reply.get("version") {
        Some(Bson::String(version)) => Ok(version.clone()),
        _ => Err(corrupt("buildInfo version")),
    }
}
