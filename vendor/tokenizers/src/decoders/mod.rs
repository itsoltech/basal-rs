// Modified for basal-rs: read the decoder tree with admitted `Replace` regex backings.
pub mod bpe;
pub mod byte_fallback;
pub mod ctc;
pub mod fuse;
pub mod sequence;
pub mod strip;
pub mod wordpiece;

// Re-export these as decoders
pub use super::pre_tokenizers::byte_level;
pub use super::pre_tokenizers::metaspace;

#[cfg(feature = "onig")]
use serde::de::{DeserializeSeed, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
#[cfg(feature = "onig")]
use std::fmt;
#[cfg(feature = "onig")]
use std::sync::Arc;

use crate::decoders::bpe::BPEDecoder;
use crate::decoders::byte_fallback::ByteFallback;
use crate::decoders::ctc::CTC;
use crate::decoders::fuse::Fuse;
use crate::decoders::sequence::Sequence;
use crate::decoders::strip::Strip;
use crate::decoders::wordpiece::WordPiece;
use crate::normalizers::replace::Replace;
#[cfg(feature = "onig")]
use crate::normalizers::replace::ReplaceDeserializer;
use crate::pre_tokenizers::byte_level::ByteLevel;
use crate::pre_tokenizers::metaspace::Metaspace;
#[cfg(feature = "onig")]
use crate::utils::merge_storage::{AdmittedRegexError, RegexCharge, TableRefusal};
use crate::{Decoder, Result};

#[derive(Serialize, Clone, Debug)]
#[serde(untagged)]
pub enum DecoderWrapper {
    BPE(BPEDecoder),
    ByteLevel(ByteLevel),
    WordPiece(WordPiece),
    Metaspace(Metaspace),
    CTC(CTC),
    Sequence(Sequence),
    Replace(Replace),
    Fuse(Fuse),
    Strip(Strip),
    ByteFallback(ByteFallback),
}

#[derive(Deserialize)]
struct Tagged {
    #[serde(rename = "type")]
    variant: EnumType,
    #[serde(flatten)]
    rest: serde_json::Value,
}
#[derive(Serialize, Deserialize)]
enum EnumType {
    BPEDecoder,
    ByteLevel,
    WordPiece,
    Metaspace,
    CTC,
    Sequence,
    Replace,
    Fuse,
    Strip,
    ByteFallback,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum DecoderHelper {
    Tagged(Tagged),
    Legacy(serde_json::Value),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum DecoderUntagged {
    BPE(BPEDecoder),
    ByteLevel(ByteLevel),
    WordPiece(WordPiece),
    Metaspace(Metaspace),
    CTC(CTC),
    Sequence(Sequence),
    Replace(Replace),
    Fuse(Fuse),
    Strip(Strip),
    ByteFallback(ByteFallback),
}

/// The tagged object with its `type` restored, as the decoder variants read it.
fn tagged_values<E: serde::de::Error>(
    model: Tagged,
) -> std::result::Result<(EnumType, serde_json::Value), E> {
    let mut values: serde_json::Map<String, serde_json::Value> =
        serde_json::from_value(model.rest).map_err(serde::de::Error::custom)?;
    values.insert(
        "type".to_string(),
        serde_json::to_value(&model.variant).map_err(serde::de::Error::custom)?,
    );
    Ok((model.variant, serde_json::Value::Object(values)))
}

fn tagged<E: serde::de::Error>(
    variant: EnumType,
    values: serde_json::Value,
) -> std::result::Result<DecoderWrapper, E> {
    Ok(match variant {
        EnumType::BPEDecoder => {
            DecoderWrapper::BPE(serde_json::from_value(values).map_err(serde::de::Error::custom)?)
        }
        EnumType::ByteLevel => DecoderWrapper::ByteLevel(
            serde_json::from_value(values).map_err(serde::de::Error::custom)?,
        ),
        EnumType::WordPiece => DecoderWrapper::WordPiece(
            serde_json::from_value(values).map_err(serde::de::Error::custom)?,
        ),
        EnumType::Metaspace => DecoderWrapper::Metaspace(
            serde_json::from_value(values).map_err(serde::de::Error::custom)?,
        ),
        EnumType::CTC => {
            DecoderWrapper::CTC(serde_json::from_value(values).map_err(serde::de::Error::custom)?)
        }
        EnumType::Sequence => DecoderWrapper::Sequence(
            serde_json::from_value(values).map_err(serde::de::Error::custom)?,
        ),
        EnumType::Replace => DecoderWrapper::Replace(
            serde_json::from_value(values).map_err(serde::de::Error::custom)?,
        ),
        EnumType::Fuse => {
            DecoderWrapper::Fuse(serde_json::from_value(values).map_err(serde::de::Error::custom)?)
        }
        EnumType::Strip => {
            DecoderWrapper::Strip(serde_json::from_value(values).map_err(serde::de::Error::custom)?)
        }
        EnumType::ByteFallback => DecoderWrapper::ByteFallback(
            serde_json::from_value(values).map_err(serde::de::Error::custom)?,
        ),
    })
}

impl<'de> Deserialize<'de> for DecoderWrapper {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let helper = DecoderHelper::deserialize(deserializer).expect("Helper");
        Ok(match helper {
            DecoderHelper::Tagged(model) => {
                let (variant, values) = tagged_values::<D::Error>(model)?;
                tagged::<D::Error>(variant, values)?
            }
            DecoderHelper::Legacy(value) => {
                let untagged = serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                match untagged {
                    DecoderUntagged::BPE(dec) => DecoderWrapper::BPE(dec),
                    DecoderUntagged::ByteLevel(dec) => DecoderWrapper::ByteLevel(dec),
                    DecoderUntagged::WordPiece(dec) => DecoderWrapper::WordPiece(dec),
                    DecoderUntagged::Metaspace(dec) => DecoderWrapper::Metaspace(dec),
                    DecoderUntagged::CTC(dec) => DecoderWrapper::CTC(dec),
                    DecoderUntagged::Sequence(dec) => DecoderWrapper::Sequence(dec),
                    DecoderUntagged::Replace(dec) => DecoderWrapper::Replace(dec),
                    DecoderUntagged::Fuse(dec) => DecoderWrapper::Fuse(dec),
                    DecoderUntagged::Strip(dec) => DecoderWrapper::Strip(dec),
                    DecoderUntagged::ByteFallback(dec) => DecoderWrapper::ByteFallback(dec),
                }
            }
        })
    }
}

/// Reads the tokenizer's optional `decoder` like `Option<DecoderWrapper>`, but
/// the regex backing of every `Replace` in the decoder tree is admitted on
/// `account`. The inner `Err` is the first typed refusal; no later decoder is
/// constructed after it. Every other variant, error and message follows the
/// legacy route.
#[cfg(feature = "onig")]
pub(crate) struct DecoderRegexSeed<E> {
    pub(crate) account: Arc<RegexCharge>,
    pub(crate) refused: fn(TableRefusal) -> E,
}

#[cfg(feature = "onig")]
impl<'de, E> DeserializeSeed<'de> for DecoderRegexSeed<E> {
    type Value = std::result::Result<Option<DecoderWrapper>, E>;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_option(self)
    }
}

#[cfg(feature = "onig")]
impl<'de, E> Visitor<'de> for DecoderRegexSeed<E> {
    type Value = std::result::Result<Option<DecoderWrapper>, E>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("option")
    }

    fn visit_none<Er: serde::de::Error>(self) -> std::result::Result<Self::Value, Er> {
        Ok(Ok(None))
    }

    fn visit_unit<Er: serde::de::Error>(self) -> std::result::Result<Self::Value, Er> {
        Ok(Ok(None))
    }

    fn visit_some<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(admitted_decoder(deserializer, &self.account)?
            .map(Some)
            .map_err(self.refused))
    }
}

/// `DecoderWrapper::deserialize` with admitted `Replace` regexes; `Sequence`
/// elements recurse in order. The outer `Ok(Err)` is a typed refusal.
#[cfg(feature = "onig")]
fn admitted_decoder<'de, D>(
    deserializer: D,
    account: &Arc<RegexCharge>,
) -> std::result::Result<std::result::Result<DecoderWrapper, TableRefusal>, D::Error>
where
    D: Deserializer<'de>,
{
    let helper = DecoderHelper::deserialize(deserializer).expect("Helper");
    Ok(Ok(match helper {
        DecoderHelper::Tagged(model) => {
            let (variant, values) = tagged_values::<D::Error>(model)?;
            match variant {
                EnumType::Replace => {
                    let replace: ReplaceDeserializer =
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?;
                    match Replace::new_admitted(replace.pattern, replace.content, account) {
                        Ok(replace) => DecoderWrapper::Replace(replace),
                        Err(AdmittedRegexError::Refused(refusal)) => return Ok(Err(refusal)),
                        // Reported as serde's `try_from` and then the variant read report it.
                        Err(AdmittedRegexError::Backend(error)) => {
                            return Err(serde::de::Error::custom(
                                <serde_json::Error as serde::de::Error>::custom(error),
                            ))
                        }
                    }
                }
                EnumType::Sequence => {
                    let sequence: sequence::shape::Sequence =
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?;
                    match admitted_sequence(sequence, account).map_err(serde::de::Error::custom)? {
                        Ok(sequence) => DecoderWrapper::Sequence(sequence),
                        Err(refusal) => return Ok(Err(refusal)),
                    }
                }
                variant => tagged::<D::Error>(variant, values)?,
            }
        }
        DecoderHelper::Legacy(value) => match admitted_untagged(&value, account) {
            Some(Ok(decoder)) => decoder,
            Some(Err(refusal)) => return Ok(Err(refusal)),
            None => {
                return Err(serde::de::Error::custom(
                    "data did not match any variant of untagged enum DecoderUntagged",
                ))
            }
        },
    }))
}

#[cfg(feature = "onig")]
fn admitted_sequence(
    sequence: sequence::shape::Sequence,
    account: &Arc<RegexCharge>,
) -> std::result::Result<std::result::Result<Sequence, TableRefusal>, serde_json::Error> {
    let mut decoders = Vec::new();
    for value in sequence.decoders {
        match admitted_decoder(value, account)? {
            Ok(decoder) => decoders.push(decoder),
            Err(refusal) => return Ok(Err(refusal)),
        }
    }
    Ok(Ok(Sequence::new(decoders)))
}

/// One variant of an untagged enum, tried as serde tries it: the input is
/// buffered first, so number and string handling match `DecoderUntagged`.
#[cfg(feature = "onig")]
#[derive(Deserialize)]
#[serde(untagged)]
enum Trial<T> {
    Variant(T),
}

#[cfg(feature = "onig")]
fn trial<T: serde::de::DeserializeOwned>(value: &serde_json::Value) -> Option<T> {
    match Trial::<T>::deserialize(value) {
        Ok(Trial::Variant(variant)) => Some(variant),
        Err(_) => None,
    }
}

/// `DecoderUntagged` in its variant order. A refusal returns at once and is
/// never a mismatch; a parse or backend error is a mismatch. `None` when no
/// variant matches.
#[cfg(feature = "onig")]
fn admitted_untagged(
    value: &serde_json::Value,
    account: &Arc<RegexCharge>,
) -> Option<std::result::Result<DecoderWrapper, TableRefusal>> {
    if let Some(decoder) = trial(value) {
        return Some(Ok(DecoderWrapper::BPE(decoder)));
    }
    if let Some(decoder) = trial(value) {
        return Some(Ok(DecoderWrapper::ByteLevel(decoder)));
    }
    if let Some(decoder) = trial(value) {
        return Some(Ok(DecoderWrapper::WordPiece(decoder)));
    }
    if let Some(decoder) = trial(value) {
        return Some(Ok(DecoderWrapper::Metaspace(decoder)));
    }
    if let Some(decoder) = trial(value) {
        return Some(Ok(DecoderWrapper::CTC(decoder)));
    }
    if let Some(sequence) = trial::<sequence::shape::Sequence>(value) {
        match admitted_sequence(sequence, account) {
            Ok(Ok(sequence)) => return Some(Ok(DecoderWrapper::Sequence(sequence))),
            Ok(Err(refusal)) => return Some(Err(refusal)),
            Err(_) => {}
        }
    }
    if let Some(replace) = trial::<ReplaceDeserializer>(value) {
        match Replace::new_admitted(replace.pattern, replace.content, account) {
            Ok(replace) => return Some(Ok(DecoderWrapper::Replace(replace))),
            Err(AdmittedRegexError::Refused(refusal)) => return Some(Err(refusal)),
            Err(AdmittedRegexError::Backend(_)) => {}
        }
    }
    if let Some(decoder) = trial(value) {
        return Some(Ok(DecoderWrapper::Fuse(decoder)));
    }
    if let Some(decoder) = trial(value) {
        return Some(Ok(DecoderWrapper::Strip(decoder)));
    }
    if let Some(decoder) = trial(value) {
        return Some(Ok(DecoderWrapper::ByteFallback(decoder)));
    }
    None
}

impl Decoder for DecoderWrapper {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        match self {
            Self::BPE(bpe) => bpe.decode_chain(tokens),
            Self::ByteLevel(bl) => bl.decode_chain(tokens),
            Self::Metaspace(ms) => ms.decode_chain(tokens),
            Self::WordPiece(wp) => wp.decode_chain(tokens),
            Self::CTC(ctc) => ctc.decode_chain(tokens),
            Self::Sequence(seq) => seq.decode_chain(tokens),
            Self::Replace(seq) => seq.decode_chain(tokens),
            Self::ByteFallback(bf) => bf.decode_chain(tokens),
            Self::Strip(bf) => bf.decode_chain(tokens),
            Self::Fuse(bf) => bf.decode_chain(tokens),
        }
    }
}

impl_enum_from!(BPEDecoder, DecoderWrapper, BPE);
impl_enum_from!(ByteLevel, DecoderWrapper, ByteLevel);
impl_enum_from!(ByteFallback, DecoderWrapper, ByteFallback);
impl_enum_from!(Fuse, DecoderWrapper, Fuse);
impl_enum_from!(Strip, DecoderWrapper, Strip);
impl_enum_from!(Metaspace, DecoderWrapper, Metaspace);
impl_enum_from!(WordPiece, DecoderWrapper, WordPiece);
impl_enum_from!(CTC, DecoderWrapper, CTC);
impl_enum_from!(Sequence, DecoderWrapper, Sequence);
impl_enum_from!(Replace, DecoderWrapper, Replace);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_serialization() {
        let oldjson = r#"{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Metaspace","replacement":"▁","add_prefix_space":true,"prepend_scheme":"always"}]}"#;
        let olddecoder: DecoderWrapper = serde_json::from_str(oldjson).unwrap();
        let oldserialized = serde_json::to_string(&olddecoder).unwrap();
        let json = r#"{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}]}"#;
        assert_eq!(oldserialized, json);

        let decoder: DecoderWrapper = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&decoder).unwrap();
        assert_eq!(serialized, json);
    }
    #[test]
    fn decoder_serialization_other_no_arg() {
        let json = r#"{"type":"Sequence","decoders":[{"type":"Fuse"},{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}]}"#;
        let decoder: DecoderWrapper = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&decoder).unwrap();
        assert_eq!(serialized, json);
    }

    #[test]
    fn decoder_serialization_no_decode() {
        let json = r#"{"type":"Sequence","decoders":[{},{"type":"Metaspace","replacement":"▁","prepend_scheme":"always"}]}"#;
        let parse = serde_json::from_str::<DecoderWrapper>(json);
        match parse {
            Err(err) => assert_eq!(
                format!("{err}"),
                "data did not match any variant of untagged enum DecoderUntagged"
            ),
            _ => panic!("Expected error"),
        }

        let json = r#"{"replacement":"▁","prepend_scheme":"always"}"#;
        let parse = serde_json::from_str::<DecoderWrapper>(json);
        match parse {
            Err(err) => assert_eq!(
                format!("{err}"),
                "data did not match any variant of untagged enum DecoderUntagged"
            ),
            _ => panic!("Expected error"),
        }

        let json = r#"{"type":"Sequence","prepend_scheme":"always"}"#;
        let parse = serde_json::from_str::<DecoderWrapper>(json);
        match parse {
            Err(err) => assert_eq!(format!("{err}"), "missing field `decoders`"),
            _ => panic!("Expected error"),
        }
    }
}
