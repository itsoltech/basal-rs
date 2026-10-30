// Modified for basal-rs: deserialize BPE with typed merge-storage refusals and
// an optionally staged forward vocabulary.
use super::{
    super::OrderedVocabIter, convert_merges_to_hashmap, BpeBuilder, BpeMergeSeed, BpeStagedTrial,
    BpeTrial, Merges, Pair, StagedVocab, Vocab, BPE,
};
use crate::utils::merge_storage::{MergeBuildError, MergeStorage};
use serde::{
    de::{DeserializeSeed, Error, MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::marker::PhantomData;

impl Serialize for BPE {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut model = serializer.serialize_struct("BPE", 8)?;

        // Start by small fields
        model.serialize_field("type", "BPE")?;
        model.serialize_field("dropout", &self.dropout)?;
        model.serialize_field("unk_token", &self.unk_token)?;
        model.serialize_field("continuing_subword_prefix", &self.continuing_subword_prefix)?;
        model.serialize_field("end_of_word_suffix", &self.end_of_word_suffix)?;
        model.serialize_field("fuse_unk", &self.fuse_unk)?;
        model.serialize_field("byte_fallback", &self.byte_fallback)?;
        model.serialize_field("ignore_merges", &self.ignore_merges)?;

        // Then the large ones
        let mut merges: Vec<(&Pair, &u32)> = self
            .merges
            .iter()
            .map(|(pair, (rank, _))| (pair, rank))
            .collect();
        merges.sort_unstable_by_key(|k| *k.1);
        let merges = merges
            .into_iter()
            .map(|(pair, _)| (self.vocab_r[&pair.0].clone(), self.vocab_r[&pair.1].clone()))
            .collect::<Vec<_>>();
        let ordered_vocab = OrderedVocabIter::new(&self.vocab_r);

        model.serialize_field("vocab", &ordered_vocab)?;
        model.serialize_field("merges", &merges)?;

        model.end()
    }
}

const FIELDS: &[&str] = &[
    "type",
    "dropout",
    "unk_token",
    "continuing_subword_prefix",
    "end_of_word_suffix",
    "fuse_unk",
    "byte_fallback",
    "ignore_merges",
    "vocab",
    "merges",
];

impl<'de> Deserialize<'de> for BPE {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let build = |builder: BpeBuilder, vocab: Vocab, merges| {
            builder.vocab_and_merges(vocab, merges).build()
        };
        deserializer.deserialize_struct("BPE", FIELDS, BPEVisitor(build, PhantomData))
    }
}

/// A storage build's refusal is the inner `Err`; a backend error stays the
/// visitor's error.
fn seed_outcome<E>(
    built: Result<BPE, MergeBuildError<E>>,
) -> crate::Result<Result<BPE, MergeBuildError<E>>> {
    match built {
        Ok(bpe) => Ok(Ok(bpe)),
        Err(MergeBuildError::Backend(error)) => Err(error),
        Err(refusal) => Ok(Err(refusal)),
    }
}

impl<'de, S: MergeStorage> DeserializeSeed<'de> for BpeMergeSeed<'_, S> {
    type Value = Result<BPE, MergeBuildError<S::Error>>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        let storage = self.storage;
        match self.forward {
            None => {
                let build = |builder: BpeBuilder, vocab: Vocab, merges| {
                    seed_outcome(
                        builder
                            .vocab_and_merges(vocab, merges)
                            .build_with_merge_storage(storage),
                    )
                };
                deserializer.deserialize_struct("BPE", FIELDS, BPEVisitor(build, PhantomData))
            }
            Some(account) => {
                let build = |builder: BpeBuilder, vocab: StagedVocab, merges| {
                    seed_outcome(builder.build_with_staged_vocab(storage, vocab, merges, account))
                };
                deserializer.deserialize_struct("BPE", FIELDS, BPEVisitor(build, PhantomData))
            }
        }
    }
}

impl<'de> Deserialize<'de> for BpeTrial {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let parsed = |builder: BpeBuilder, vocab: Vocab, merges| {
            Ok(BpeTrial(builder.vocab_and_merges(vocab, merges)))
        };
        deserializer.deserialize_struct("BPE", FIELDS, BPEVisitor(parsed, PhantomData))
    }
}

impl<'de> Deserialize<'de> for BpeStagedTrial {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let parsed = |builder, vocab, merges| {
            Ok(BpeStagedTrial {
                builder,
                vocab,
                merges,
            })
        };
        deserializer.deserialize_struct("BPE", FIELDS, BPEVisitor(parsed, PhantomData))
    }
}

/// serde_core 1.0.229's `size_hint::cautious::<(String, u32)>`
/// (`private/size_hint.rs`), the capacity its `HashMap` visitor requests.
/// The element size is nonzero. Re-check on any serde lock change.
fn cautious_vocab_capacity(hint: Option<usize>) -> usize {
    const MAX_PREALLOC_BYTES: usize = 1024 * 1024;
    hint.unwrap_or(0)
        .min(MAX_PREALLOC_BYTES / std::mem::size_of::<(String, u32)>())
}

impl<'de> Deserialize<'de> for StagedVocab {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(StagedVocabVisitor)
    }
}

/// serde_core 1.0.229's `HashMap` visitor (`de/impls.rs`) with
/// `ahash::RandomState` and `(String, u32)` entries, except that each entry is
/// staged instead of inserted.
struct StagedVocabVisitor;
impl<'de> Visitor<'de> for StagedVocabVisitor {
    type Value = StagedVocab;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.write_str("a map")
    }

    /// The same calls in the same order: `size_hint`, the cautious capacity,
    /// one `RandomState::new()` (serde's `S::default()`), then `next_entry`
    /// until the end or the first error.
    fn visit_map<A>(self, mut map: A) -> Result<StagedVocab, A::Error>
    where
        A: MapAccess<'de>,
    {
        let capacity = cautious_vocab_capacity(map.size_hint());
        let hasher = ahash::RandomState::new();
        let mut entries = Vec::with_capacity(capacity);
        while let Some(entry) = map.next_entry::<String, u32>()? {
            entries.push(entry);
        }
        Ok(StagedVocab {
            capacity,
            hasher,
            entries,
        })
    }
}

/// Parses the fields, then hands the configured builder, the `vocab` field as
/// `Voc` and the converted merges to `F`.
struct BPEVisitor<Voc, F>(F, PhantomData<Voc>);
impl<'de, T, Voc, F> Visitor<'de> for BPEVisitor<Voc, F>
where
    Voc: Deserialize<'de>,
    F: FnOnce(BpeBuilder, Voc, Merges) -> crate::Result<T>,
{
    type Value = T;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct BPE")
    }

    fn visit_map<V>(self, mut map: V) -> std::result::Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut builder = BpeBuilder::new();
        let mut vocab: Option<Voc> = None;

        #[derive(Debug, Deserialize)]
        #[serde(untagged)]
        enum MergeType {
            Tuple(Vec<(String, String)>),
            Legacy(Vec<String>),
        }
        let mut merges: Option<MergeType> = None;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "dropout" => {
                    if let Some(dropout) = map.next_value()? {
                        builder = builder.dropout(dropout);
                    }
                }
                "unk_token" => {
                    if let Some(unk) = map.next_value()? {
                        builder = builder.unk_token(unk);
                    }
                }
                "continuing_subword_prefix" => {
                    if let Some(prefix) = map.next_value()? {
                        builder = builder.continuing_subword_prefix(prefix);
                    }
                }
                "end_of_word_suffix" => {
                    if let Some(suffix) = map.next_value()? {
                        builder = builder.end_of_word_suffix(suffix);
                    }
                }
                "fuse_unk" => {
                    if let Some(suffix) = map.next_value()? {
                        builder = builder.fuse_unk(suffix);
                    }
                }
                "byte_fallback" => {
                    if let Some(suffix) = map.next_value()? {
                        builder = builder.byte_fallback(suffix);
                    }
                }
                "ignore_merges" => {
                    if let Some(suffix) = map.next_value()? {
                        builder = builder.ignore_merges(suffix);
                    }
                }
                "vocab" => vocab = Some(map.next_value()?),
                "merges" => merges = Some(map.next_value()?),
                "type" => match map.next_value()? {
                    "BPE" => {}
                    u => {
                        return Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(u),
                            &"BPE",
                        ))
                    }
                },
                _ => {}
            }
        }
        if let (Some(vocab), Some(merges)) = (vocab, merges) {
            let merges = match merges {
                MergeType::Tuple(merges) => merges,
                MergeType::Legacy(merges) => {
                    convert_merges_to_hashmap(merges.into_iter()).map_err(Error::custom)?
                }
            };
            Ok((self.0)(builder, vocab, merges).map_err(Error::custom)?)
        } else {
            Err(Error::custom("Missing vocab/merges"))
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::models::bpe::Vocab;

    #[test]
    fn test_serialization() {
        let vocab: Vocab = [
            ("<unk>".into(), 0),
            ("a".into(), 1),
            ("b".into(), 2),
            ("ab".into(), 3),
        ]
        .iter()
        .cloned()
        .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![("a".to_string(), "b".to_string())])
            .unk_token("<unk>".to_string())
            .ignore_merges(true)
            .build()
            .unwrap();

        let legacy = r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":["a b"]}"#;
        let legacy = serde_json::from_str(legacy).unwrap();
        assert_eq!(bpe, legacy);

        let data = serde_json::to_string(&bpe).unwrap();
        assert_eq!(
            data,
            r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":[["a","b"]]}"#
        );
        let reconstructed = serde_json::from_str(&data).unwrap();
        assert_eq!(bpe, reconstructed);

        // With a space in the token
        let vocab: Vocab = [
            ("<unk>".into(), 0),
            ("a".into(), 1),
            ("b c d".into(), 2),
            ("ab c d".into(), 3),
        ]
        .iter()
        .cloned()
        .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![("a".to_string(), "b c d".to_string())])
            .unk_token("<unk>".to_string())
            .ignore_merges(true)
            .build()
            .unwrap();
        let data = serde_json::to_string(&bpe).unwrap();
        assert_eq!(
            data,
            r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b c d":2,"ab c d":3},"merges":[["a","b c d"]]}"#
        );
        let reconstructed = serde_json::from_str(&data).unwrap();
        assert_eq!(bpe, reconstructed);
    }

    #[test]
    fn test_serialization_ignore_merges() {
        let vocab: Vocab = [("<unk>".into(), 0), ("a".into(), 1), ("b".into(), 2)]
            .iter()
            .cloned()
            .collect();
        let mut bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![])
            .unk_token("<unk>".to_string())
            .ignore_merges(true)
            .build()
            .unwrap();

        let bpe_string = r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2},"merges":[]}"#;
        assert_eq!(serde_json::from_str::<BPE>(bpe_string).unwrap(), bpe);

        bpe.ignore_merges = false;
        let bpe_string = r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"vocab":{"<unk>":0,"a":1,"b":2},"merges":[]}"#;
        assert_eq!(serde_json::from_str::<BPE>(bpe_string).unwrap(), bpe);
    }
}
