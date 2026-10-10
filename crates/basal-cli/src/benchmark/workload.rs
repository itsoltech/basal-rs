//! Reproducible HTTP workloads; original public questions and synthetic mixed traffic stay distinct.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use basal_core::{Backend, DecideError, Engine};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const SET_SHA256: &str = "5b701fe861d626b3983928d3e5e168e452bbf6ba4137d75c76597e71e7691291";

/// Default workloads compiled into the binary, written by `tools/decision-sets/build_benchmark.py --profile v2`.
/// The hashes repeat `tools/decision-sets/benchmark-v2.lock.json`; the files are immutable, a change is a new profile.
const PROFILE: &str = "public_900_and_synthetic_mixed_v2";
const MANIFEST: &str = include_str!("../../benchmark-data/v2/manifest.json");
const MANIFEST_SHA256: &str = "a645db4c5e880eda6aa4f10b822bc9ea69a4a44699186152f276c377d54dec9f";
const SEQUENTIAL: &str = include_str!("../../benchmark-data/v2/sequential.jsonl");
const SEQUENTIAL_SHA256: &str = "81756119f5fe3a4e44aefa1c1cc56e5cd3153001141fb715ec502bb3ee193374";
const MIXED: &str = include_str!("../../benchmark-data/v2/mixed.jsonl");
const MIXED_SHA256: &str = "bd6400e6e7392b274af21c73914fb334bbca02f1a00dcd4cfd86716e300007fe";

pub(super) struct RawRequest {
    id: String,
    class: String,
    auto_class: bool,
    body: Value,
}

pub(super) struct RawWorkload {
    pub source: Value,
    pub sequential: Vec<RawRequest>,
    pub mixed: Vec<RawRequest>,
}

pub(super) struct Request {
    pub id: String,
    pub class: String,
    pub body: Arc<Vec<u8>>,
    pub key: [u8; 32],
    pub questions: usize,
    pub input_tokens: usize,
}

pub(super) type Workload = Vec<Arc<Request>>;

fn records(s: &str) -> Result<Vec<Value>> {
    s.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).context("request JSONL")).collect()
}

fn size_class(bytes: usize) -> &'static str {
    match bytes {
        0..=2048 => "payload-le-2KiB",
        2049..=16384 => "payload-2-to-16KiB",
        _ => "payload-gt-16KiB",
    }
}

fn replay(records: Vec<Value>) -> Result<Vec<RawRequest>> {
    records
        .into_iter()
        .enumerate()
        .map(|(i, r)| {
            let body = r.get("request").context("each line needs request")?.clone();
            ensure!(body.is_object(), "request {i}: request must be an object");
            let class = r["class"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| size_class(body.to_string().len()).to_string());
            Ok(RawRequest {
                id: r["id"].as_str().map(str::to_owned).unwrap_or_else(|| format!("request-{i}")),
                class,
                auto_class: r["class"].as_str().is_none(),
                body,
            })
        })
        .collect()
}

pub(super) fn load(requests: Option<&Path>) -> Result<RawWorkload> {
    if let Some(path) = requests {
        let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let records = records(std::str::from_utf8(&data)?)?;
        ensure!(!records.is_empty() && records.len() <= 100_000, "--requests needs 1..100000 requests");
        return Ok(RawWorkload {
            source: json!({"kind": "request_replay", "sha256": format!("{:x}", Sha256::digest(&data)), "requests": records.len()}),
            sequential: replay(records.clone())?,
            mixed: replay(records)?,
        });
    }
    let hash = format!("{:x}", Sha256::digest(MANIFEST));
    ensure!(hash == MANIFEST_SHA256, "embedded benchmark manifest SHA-256 {hash}; expected {MANIFEST_SHA256}");
    let manifest: Value = serde_json::from_str(MANIFEST).context("embedded benchmark manifest")?;
    ensure!(
        manifest["schema_version"] == 2
            && manifest["profile"] == PROFILE
            && manifest["public_set_sha256"] == SET_SHA256,
        "embedded benchmark manifest does not describe {PROFILE}"
    );
    Ok(RawWorkload {
        sequential: embedded(&manifest, "sequential.jsonl", SEQUENTIAL, SEQUENTIAL_SHA256)?,
        mixed: embedded(&manifest, "mixed.jsonl", MIXED, MIXED_SHA256)?,
        source: json!({"kind": "embedded", "profile": PROFILE, "manifest_sha256": MANIFEST_SHA256,
            "public_set_sha256": SET_SHA256, "workloads": manifest["workloads"]}),
    })
}

/// An embedded workload must match its pinned hash and the manifest counts, so the profile cannot change silently.
fn embedded(manifest: &Value, name: &str, data: &str, sha256: &str) -> Result<Vec<RawRequest>> {
    let hash = format!("{:x}", Sha256::digest(data));
    let expected = &manifest["workloads"][name];
    ensure!(hash == sha256 && expected["sha256"] == sha256, "embedded {name}: SHA-256 {hash}; expected {sha256}");
    let records = records(data)?;
    let mut classes: BTreeMap<&str, usize> = BTreeMap::new();
    let mut ids = BTreeSet::new();
    let mut questions = 0;
    for r in &records {
        *classes.entry(r["class"].as_str().unwrap_or("auto-payload-size")).or_default() += 1;
        ids.insert(r["id"].as_str().context("embedded request id")?);
        questions += r["request"]["questions"].as_object().context("embedded request questions")?.len();
    }
    ensure!(
        expected["requests"] == records.len()
            && ids.len() == records.len()
            && expected["questions"] == questions
            && expected["classes"] == json!(classes),
        "embedded {name}: requests, IDs, questions or classes differ from the manifest"
    );
    replay(records)
}

pub(super) fn prepare<B: Backend>(
    raw: Vec<RawRequest>,
    engine: &Engine<B>,
    shuffle: bool,
) -> Result<(Workload, Value)> {
    // Repeated synthetic requests share their encoded payload and token metadata.
    let mut pool: BTreeMap<[u8; 32], Arc<Request>> = BTreeMap::new();
    let mut out = Vec::with_capacity(raw.len());
    let mut skipped = Vec::new();
    for mut r in raw {
        r.body["model"] = json!(engine.manifest.name);
        let body = serde_json::to_vec(&r.body)?;
        if r.auto_class {
            r.class = size_class(body.len()).to_string();
        }
        let key: [u8; 32] = Sha256::digest(&body).into();
        let request = if let Some(p) = pool.get(&key) {
            ensure!(p.body.as_ref() == &body, "request digest collision");
            Arc::new(Request {
                id: r.id,
                class: r.class,
                body: p.body.clone(),
                key: p.key,
                questions: p.questions,
                input_tokens: p.input_tokens,
            })
        } else {
            let plan = match engine.plan_request(&r.body) {
                Ok(plan) => plan,
                Err(DecideError::Unsupported(errors))
                    if !errors.is_empty() && errors.iter().all(|e| e.kind == "context_length_exceeded") =>
                {
                    // Keep the whole request intact: removing individual questions would change the workload.
                    skipped.push(json!({"id": r.id, "class": r.class, "detail": errors}));
                    continue;
                }
                Err(error) => return Err(error).with_context(|| format!("planning benchmark request {}", r.id)),
            };
            let input_tokens = plan.prepared.iter().flat_map(|p| &p.orders).map(|o| o.input_ids.len()).sum();
            let questions = r.body["questions"].as_object().context("questions object")?.len();
            let p = Arc::new(Request { id: r.id, class: r.class, body: Arc::new(body), key, questions, input_tokens });
            pool.insert(key, p.clone());
            p
        };
        out.push(request);
    }
    if shuffle {
        out.sort_by_cached_key(|r| Sha256::digest(format!("mixed-v1/7/{}", r.id)).to_vec());
    }
    let description = description(&out, &skipped);
    Ok((out, description))
}

fn description(requests: &Workload, skipped: &[Value]) -> Value {
    let mut classes: BTreeMap<&str, usize> = BTreeMap::new();
    for r in requests {
        *classes.entry(&r.class).or_default() += 1;
    }
    let mut hash = Sha256::new();
    let unique = requests.iter().map(|r| r.key).collect::<std::collections::BTreeSet<_>>().len();
    for r in requests {
        hash.update(r.key);
    }
    json!({"source_requests": requests.len() + skipped.len(), "requests": requests.len(),
        "skipped_requests": skipped.len(), "skipped": skipped,
        "unique_payloads": unique, "classes": classes, "ordered_payloads_sha256": format!("{:x}", hash.finalize()),
        "payload_bytes": super::latency(&requests.iter().map(|r| r.body.len() as f64).collect::<Vec<_>>()),
        "input_tokens_both_orders": super::latency(&requests.iter().map(|r| r.input_tokens as f64).collect::<Vec<_>>())})
}
