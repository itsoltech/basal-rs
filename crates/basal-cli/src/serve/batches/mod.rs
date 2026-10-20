//! Durable Batch and Files API of `basal serve` (disabled by default).
//!
//! [`BatchService`] owns the batch state of one process: the store connection, the artifact store, the
//! authenticator, the CPU planning threads, the execution pool `P` and the supervisors (discovery and validation,
//! lifecycle: expiry, drain, settlement, finalization, GC, cleanup, retention and recovery). It runs no GPU work;
//! the background executor (`executor.rs`, `bg_lane.rs`, `scheduler.rs`, `device_budget.rs`) uses its store,
//! artifacts, profiles, planning threads and host pool.
//!
//! Online inference is independent of it: a database or storage outage makes only the Batch/Files routes answer
//! 503, and `/health` keeps its GPU semantics (`/health/batches` reports this service).

pub(crate) mod artifacts;
pub(crate) mod auth;
pub(crate) mod config;
pub(crate) mod dto;
pub(crate) mod store;

mod api;
pub(super) mod bg_lane;
pub(crate) mod budget;
mod convert;
pub(super) mod device_budget;
pub(super) mod executor;
mod ingest;
mod lease;
mod lifecycle;
pub(crate) mod planning;
mod profile;
mod scheduler;
mod service;
mod upload;
mod validation;

pub(crate) use profile::{ModelProfile, ResolvedProfile};
pub(crate) use service::{connect_store, BatchService};
