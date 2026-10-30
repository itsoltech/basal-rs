//! Discover one dense checkpoint or the files named by a bounded Hugging Face safetensors index, and read an
//! operator-declared BF16 checkpoint through one retained handle per file ([`Inventory`]).
use anyhow::{bail, ensure, Context, Result};
use serde::de::{Deserialize, Deserializer, IgnoredAny, MapAccess, Visitor};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};

/// Most shards of an indexed checkpoint.
pub const MAX_SHARDS: usize = 64;
/// Largest checkpoint index file.
pub const MAX_INDEX_BYTES: u64 = 16 << 20;
/// Largest safetensors header of one file (the limit of the safetensors crate).
pub const MAX_HEADER_BYTES: u64 = 100_000_000;
/// Largest sum of the safetensors headers of one [`Inventory`].
pub const MAX_TOTAL_HEADER_BYTES: u64 = 256 << 20;
/// Largest [`Inventory`] (each file and the sum of its files).
pub const MAX_CHECKPOINT_BYTES: u64 = 1 << 40;

/// Checkpoint files in deterministic filename order. Index entries must be plain safetensors filenames.
/// HF snapshot symlinks remain usable; an index cannot introduce a path outside the supplied directory.
pub fn files(dir: &Path) -> Result<Vec<PathBuf>> {
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }
    let index = dir.join("model.safetensors.index.json");
    let file = std::fs::File::open(&index).with_context(|| format!("opening {}", index.display()))?;
    ensure!(file.metadata()?.len() <= MAX_INDEX_BYTES, "checkpoint index exceeds 16 MiB");
    let value: Value = serde_json::from_reader(file).context("checkpoint index JSON")?;
    let weights = value
        .get("weight_map")
        .and_then(Value::as_object)
        .filter(|m| !m.is_empty())
        .context("checkpoint index needs a nonempty weight_map")?;
    let mut names = std::collections::BTreeSet::new();
    for value in weights.values() {
        let name = value.as_str().context("checkpoint shard name must be a string")?;
        ensure!(
            name.ends_with(".safetensors")
                && !name.starts_with('.')
                && name.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.')),
            "checkpoint shard must be a plain safetensors filename"
        );
        names.insert(name.to_string());
        ensure!(names.len() <= MAX_SHARDS, "at most 64 checkpoint shards");
    }
    names
        .into_iter()
        .map(|name| {
            let path = dir.join(name);
            ensure!(path.is_file(), "checkpoint shard {} missing", path.display());
            Ok(path)
        })
        .collect()
}

/// Actual bytes on disk, including every shard's header; suitable for memory/download diagnostics.
pub fn bytes(dir: &Path) -> Result<u64> {
    files(dir)?
        .iter()
        .try_fold(0u64, |sum, p| sum.checked_add(p.metadata()?.len()).context("checkpoint byte count overflow"))
}

/// A file name usable inside one directory: 1 to 255 bytes of `[A-Za-z0-9_.-]`, not starting with `.`.
pub fn is_plain_file_name(name: &str) -> bool {
    (1..=255).contains(&name.len())
        && !name.starts_with('.')
        && name.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
}

/// A JSON object read with every entry kept, so that a repeated key is refused by [`UniqueMap::into_map`] instead
/// of replacing an earlier entry (as `serde_json::Value`, `HashMap` and `#[serde(flatten)]` do).
pub struct UniqueMap<V>(Vec<(String, V)>);

impl<V> UniqueMap<V> {
    /// The entries by key; `what` names the object in the error of a repeated key.
    pub fn into_map(self, what: &str) -> Result<BTreeMap<String, V>> {
        let mut map = BTreeMap::new();
        for (key, value) in self.0 {
            match map.entry(key) {
                std::collections::btree_map::Entry::Vacant(e) => {
                    e.insert(value);
                }
                std::collections::btree_map::Entry::Occupied(e) => bail!("duplicate key {} in {what}", e.key()),
            }
        }
        Ok(map)
    }
}

impl<'de, V: Deserialize<'de>> Deserialize<'de> for UniqueMap<V> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Entries<V>(std::marker::PhantomData<V>);
        impl<'de, V: Deserialize<'de>> Visitor<'de> for Entries<V> {
            type Value = UniqueMap<V>;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut entries = Vec::new();
                while let Some(entry) = map.next_entry::<String, V>()? {
                    entries.push(entry);
                }
                Ok(UniqueMap(entries))
            }
        }
        deserializer.deserialize_map(Entries(std::marker::PhantomData))
    }
}

/// `model.safetensors.index.json` as read for an [`Inventory`]: unknown fields and repeated keys are refused.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexFile {
    weight_map: UniqueMap<String>,
    #[serde(rename = "metadata")]
    _metadata: Option<IgnoredAny>,
}

/// A checkpoint index parsed from its exact bytes: the shard filename of every tensor.
pub struct CheckpointIndex {
    weight_map: BTreeMap<String, String>,
}

impl CheckpointIndex {
    /// Parse strict UTF-8 JSON of at most 16 MiB; the shards must be 1 to 64 plain `.safetensors` filenames.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        ensure!(u64::try_from(bytes.len())? <= MAX_INDEX_BYTES, "checkpoint index exceeds 16 MiB");
        let text = std::str::from_utf8(bytes).context("checkpoint index is not UTF-8")?;
        let file: IndexFile = serde_json::from_str(text).context("checkpoint index JSON")?;
        let index = Self { weight_map: file.weight_map.into_map("checkpoint index weight_map")? };
        let shards = index.shards();
        ensure!(!shards.is_empty(), "checkpoint index needs a nonempty weight_map");
        for name in &shards {
            ensure!(
                name.ends_with(".safetensors") && is_plain_file_name(name),
                "checkpoint shard must be a plain safetensors filename"
            );
        }
        ensure!(shards.len() <= MAX_SHARDS, "at most 64 checkpoint shards");
        Ok(index)
    }

    /// The distinct shard filenames in byte order.
    pub fn shards(&self) -> Vec<&str> {
        self.weight_map.values().map(String::as_str).collect::<BTreeSet<_>>().into_iter().collect()
    }
}

/// File identity from `fstat`/`stat`; equality of two stamps is the stable-directory fence.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Stamp {
    dev: u64,
    ino: u64,
    len: u64,
    mtime: i64,
    mtime_nsec: i64,
    ctime: i64,
    ctime_nsec: i64,
}

impl Stamp {
    pub fn of(meta: &std::fs::Metadata) -> Self {
        Self {
            dev: meta.dev(),
            ino: meta.ino(),
            len: meta.len(),
            mtime: meta.mtime(),
            mtime_nsec: meta.mtime_nsec(),
            ctime: meta.ctime(),
            ctime_nsec: meta.ctime_nsec(),
        }
    }

    /// Size in bytes when stamped.
    pub fn bytes(&self) -> u64 {
        self.len
    }

    /// `path` (symlinks followed) still has this stamp; `name` labels the error.
    pub fn check(&self, path: &Path, name: &str) -> Result<()> {
        ensure!(
            std::fs::metadata(path).is_ok_and(|m| Stamp::of(&m) == *self),
            "{name} changed while loading thinking (stable-directory contract)"
        );
        Ok(())
    }
}

/// One entry of a safetensors header, before validation.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

/// The top-level entries of a safetensors header in file order, repeated keys kept; `None` is `__metadata__`.
struct RawHeader(Vec<(String, Option<RawEntry>)>);

impl<'de> Deserialize<'de> for RawHeader {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Entries;
        impl<'de> Visitor<'de> for Entries {
            type Value = RawHeader;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a safetensors header object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut entries = Vec::new();
                while let Some(key) = map.next_key::<String>()? {
                    if key == "__metadata__" {
                        map.next_value::<IgnoredAny>()?;
                        entries.push((key, None));
                    } else {
                        let entry = map.next_value::<RawEntry>()?;
                        entries.push((key, Some(entry)));
                    }
                }
                Ok(RawHeader(entries))
            }
        }
        deserializer.deserialize_map(Entries)
    }
}

/// A routed BF16 tensor: its shape and its byte range in the data section of its shard.
pub struct TensorEntry {
    pub shape: Vec<usize>,
    shard: usize,
    start: u64,
    end: u64,
}

struct Shard {
    name: String,
    file: File,
    /// first byte of the data section
    base: u64,
    stamp: Stamp,
    header_bytes: u64,
}

/// One retained handle per safetensors file of a checkpoint, used for its header, content hash, every tensor read
/// and the fence; no file is opened twice or mapped. Invariants after [`Inventory::open`]: every dtype is BF16; the
/// tensors of each shard cover its data section contiguously from offset 0, each with exactly its shape's bytes;
/// tensor names are unique within and across shards; with an index the routing equals its `weight_map`; at most
/// `max_tensors` tensors. The handles close when the inventory drops.
pub struct Inventory {
    dir: PathBuf,
    shards: Vec<Shard>,
    routing: BTreeMap<String, TensorEntry>,
}

impl Inventory {
    /// Open `shards` (plain filename and declared bytes) of `dir` once each and validate their headers; `index` is
    /// the parsed index of an indexed checkpoint (None: one file). Nothing is hashed here ([`Inventory::sha256`]).
    pub fn open(
        dir: &Path,
        shards: &[(&str, u64)],
        index: Option<&CheckpointIndex>,
        max_tensors: usize,
    ) -> Result<Self> {
        ensure!(!shards.is_empty(), "a checkpoint needs a safetensors file");
        ensure!(shards.len() <= MAX_SHARDS, "at most 64 checkpoint shards");
        let mut inventory = Self { dir: dir.to_path_buf(), shards: Vec::new(), routing: BTreeMap::new() };
        let (mut total, mut headers) = (0u64, 0u64);
        for &(name, declared) in shards {
            ensure!(
                name.ends_with(".safetensors") && is_plain_file_name(name),
                "checkpoint shard must be a plain safetensors filename"
            );
            let path = dir.join(name);
            let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
            let meta = file.metadata().with_context(|| format!("{name}: fstat"))?;
            ensure!(meta.is_file(), "{name} is not a regular file");
            let len = meta.len();
            ensure!(len == declared, "{name}: {len} bytes on disk, the descriptor declares {declared}");
            total =
                total.checked_add(len).filter(|t| *t <= MAX_CHECKPOINT_BYTES).context("checkpoint exceeds 1 TiB")?;
            let mut n = [0u8; 8];
            ensure!(len >= 8, "invalid safetensors header length in {name}");
            file.read_exact_at(&mut n, 0).with_context(|| format!("{name}: reading the header length"))?;
            let n = u64::from_le_bytes(n);
            ensure!(
                n <= MAX_HEADER_BYTES && n.checked_add(8).is_some_and(|end| end <= len),
                "invalid safetensors header length {n} in {name}"
            );
            headers = headers
                .checked_add(n)
                .filter(|h| *h <= MAX_TOTAL_HEADER_BYTES)
                .context("safetensors headers exceed 256 MiB in total")?;
            let RawHeader(entries) = {
                let mut header = vec![0u8; usize::try_from(n)?];
                file.read_exact_at(&mut header, 8).with_context(|| format!("{name}: reading the header"))?;
                let text = std::str::from_utf8(&header).with_context(|| format!("{name}: header is not UTF-8"))?;
                serde_json::from_str(text).with_context(|| format!("{name}: safetensors header"))?
            };
            let mut keys = BTreeSet::new();
            for (key, _) in &entries {
                ensure!(keys.insert(key.as_str()), "duplicate key {key} in safetensors header of {name}");
            }
            drop(keys);
            let mut tensors = Vec::new();
            for (key, entry) in entries {
                let Some(entry) = entry else { continue };
                ensure!(key.len() <= 256, "{name}: tensor name exceeds 256 bytes");
                ensure!(entry.shape.len() <= 8, "{name}: tensor {key}: shape rank exceeds 8");
                ensure!(
                    inventory.routing.len() + tensors.len() < max_tensors,
                    "checkpoint has more than {max_tensors} tensors"
                );
                ensure!(entry.dtype == "BF16", "{name}: tensor {key}: dtype {}, expected BF16", entry.dtype);
                let [start, end] = entry.data_offsets;
                tensors.push((start, end, key, entry.shape));
            }
            tensors.sort();
            let mut covered = 0u64;
            for (start, end, key, shape) in &tensors {
                ensure!(*start == covered && start <= end, "{name}: tensor offsets are not contiguous at {key}");
                let need = shape
                    .iter()
                    .try_fold(2u64, |bytes, &d| bytes.checked_mul(u64::try_from(d).ok()?))
                    .with_context(|| format!("{name}: tensor {key}: shape {shape:?} overflows"))?;
                ensure!(end - start == need, "{name}: tensor {key}: {} bytes, shape needs {need}", end - start);
                covered = *end;
            }
            ensure!(covered.checked_add(8 + n) == Some(len), "{name}: data section not fully covered");
            let shard = inventory.shards.len();
            for (start, end, key, shape) in tensors {
                if let Some(other) = inventory.routing.get(&key) {
                    bail!("tensor {key} is stored in both {} and {name}", inventory.shards[other.shard].name);
                }
                inventory.routing.insert(key, TensorEntry { shape, shard, start, end });
            }
            inventory.shards.push(Shard {
                name: name.to_string(),
                file,
                base: 8 + n,
                stamp: Stamp::of(&meta),
                header_bytes: n,
            });
        }
        match index {
            Some(index) => {
                let opened: BTreeSet<&str> = inventory.shards.iter().map(|s| s.name.as_str()).collect();
                ensure!(opened == index.shards().into_iter().collect(), "checkpoint shards differ from the index");
                for (key, shard) in &index.weight_map {
                    let entry = inventory
                        .routing
                        .get(key)
                        .with_context(|| format!("index lists {key}, absent from every shard"))?;
                    let stored = &inventory.shards[entry.shard].name;
                    ensure!(stored == shard, "index routes {key} to {shard}, stored in {stored}");
                }
                for (key, entry) in &inventory.routing {
                    ensure!(
                        index.weight_map.contains_key(key),
                        "shard {} stores {key}, absent from the index",
                        inventory.shards[entry.shard].name
                    );
                }
            }
            None => ensure!(inventory.shards.len() == 1, "a checkpoint without an index is one file"),
        }
        Ok(inventory)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.routing.get(name)
    }

    /// Tensor names in byte order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.routing.keys().map(String::as_str)
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn tensor_count(&self) -> usize {
        self.routing.len()
    }

    /// Sum of the file sizes.
    pub fn total_bytes(&self) -> u64 {
        self.shards.iter().map(|s| s.stamp.bytes()).sum()
    }

    /// Largest safetensors header (bytes after the length prefix).
    pub fn max_header_bytes(&self) -> u64 {
        self.shards.iter().map(|s| s.header_bytes).max().unwrap_or(0)
    }

    /// Sum of the safetensors headers.
    pub fn total_header_bytes(&self) -> u64 {
        self.shards.iter().map(|s| s.header_bytes).sum()
    }

    /// Append the bytes of tensor `name` to `out`, read through its shard's handle.
    pub fn read(&self, name: &str, out: &mut Vec<u8>) -> Result<()> {
        let entry = self.routing.get(name).with_context(|| format!("tensor {name} not in the checkpoint"))?;
        let shard = &self.shards[entry.shard];
        let at = out.len();
        let len = at.checked_add(usize::try_from(entry.end - entry.start)?).context("tensor size overflows")?;
        out.resize(len, 0);
        shard
            .file
            .read_exact_at(&mut out[at..], shard.base + entry.start)
            .with_context(|| format!("reading tensor {name} from {}", shard.name))?;
        Ok(())
    }

    /// Lowercase hex SHA-256 of the whole file `shard` read through its retained handle in 1 MiB chunks; a byte past
    /// the stamped size is a change of the file.
    pub fn sha256(&self, shard: &str) -> Result<String> {
        use sha2::{Digest, Sha256};
        let s = self.shards.iter().find(|s| s.name == shard).with_context(|| format!("{shard} is not a shard"))?;
        let len = s.stamp.bytes();
        let mut hash = Sha256::new();
        let mut buffer = vec![0u8; 1 << 20];
        let mut at = 0u64;
        while at < len {
            let n = usize::try_from((len - at).min(1 << 20))?;
            s.file.read_exact_at(&mut buffer[..n], at).with_context(|| format!("hashing {shard}"))?;
            hash.update(&buffer[..n]);
            at += n as u64;
        }
        ensure!(
            s.file.read_at(&mut buffer[..1], len)? == 0,
            "{shard} changed while loading thinking (stable-directory contract)"
        );
        Ok(format!("{:x}", hash.finalize()))
    }

    /// Every handle (`fstat`) and every `dir/name` (`stat`, symlinks followed) still has its first stamp.
    pub fn check_unchanged(&self) -> Result<()> {
        for s in &self.shards {
            ensure!(
                s.file.metadata().is_ok_and(|m| Stamp::of(&m) == s.stamp),
                "{} changed while loading thinking (stable-directory contract)",
                s.name
            );
            s.stamp.check(&self.dir.join(&s.name), &s.name)?;
        }
        Ok(())
    }
}
