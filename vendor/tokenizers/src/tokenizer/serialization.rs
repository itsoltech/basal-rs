use std::cell::Cell;
use std::convert::Infallible;
use std::marker::PhantomData;

use serde::{
    self,
    de::{DeserializeSeed, Error, IgnoredAny, MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};

use super::{added_vocabulary::AddedTokenWithId, TokenizerImpl};
use crate::models::ModelMergeSeed;
use crate::utils::merge_storage::{MergeBuildError, MergeStorage};
use crate::{
    Decoder, DecoderWrapper, Model, ModelWrapper, Normalizer, NormalizerWrapper, PostProcessor,
    PostProcessorWrapper, PreTokenizer, PreTokenizerWrapper, TokenizerBuilder,
};

static SERIALIZATION_VERSION: &str = "1.0";

impl<M, N, PT, PP, D> Serialize for TokenizerImpl<M, N, PT, PP, D>
where
    M: Serialize,
    N: Serialize,
    PT: Serialize,
    PP: Serialize,
    D: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut tokenizer = serializer.serialize_struct("Tokenizer", 9)?;

        // Start by adding the current version
        tokenizer.serialize_field("version", SERIALIZATION_VERSION)?;

        // Params
        tokenizer.serialize_field("truncation", &self.truncation)?;
        tokenizer.serialize_field("padding", &self.padding)?;

        // Added tokens
        tokenizer.serialize_field("added_tokens", &self.added_vocabulary)?;

        // Then add our parts
        tokenizer.serialize_field("normalizer", &self.normalizer)?;
        tokenizer.serialize_field("pre_tokenizer", &self.pre_tokenizer)?;
        tokenizer.serialize_field("post_processor", &self.post_processor)?;
        tokenizer.serialize_field("decoder", &self.decoder)?;
        tokenizer.serialize_field("model", &self.model)?;

        tokenizer.end()
    }
}

impl<'de, M, N, PT, PP, D> Deserialize<'de> for TokenizerImpl<M, N, PT, PP, D>
where
    M: Deserialize<'de> + Model,
    N: Deserialize<'de> + Normalizer,
    PT: Deserialize<'de> + PreTokenizer,
    PP: Deserialize<'de> + PostProcessor,
    D: Deserialize<'de> + Decoder,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        match deserializer.deserialize_struct(
            "Tokenizer",
            FIELDS,
            TokenizerVisitor::new(PlainModel),
        )? {
            Ok(tokenizer) => Ok(tokenizer),
            Err(never) => match never {},
        }
    }
}

const FIELDS: &[&str] = &[
    "version",
    "truncation",
    "padding",
    "added_tokens",
    "normalizer",
    "pre_tokenizer",
    "post_processor",
    "decoder",
    "model",
];

type WrappedTokenizer = TokenizerImpl<
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

/// Deserializes a tokenizer whose BPE model takes its merge scratch from
/// explicit storage. Storage refusal is the inner `Err`, or is left in
/// `refused` when a serde error ends the remaining map after the refusal.
pub(crate) struct TokenizerMergeSeed<'a, S: MergeStorage> {
    pub(crate) storage: &'a S,
    pub(crate) refused: &'a Cell<Option<MergeBuildError<S::Error>>>,
}

impl<'de, S: MergeStorage> DeserializeSeed<'de> for TokenizerMergeSeed<'_, S> {
    type Value = Result<WrappedTokenizer, MergeBuildError<S::Error>>;

    fn deserialize<De>(self, deserializer: De) -> Result<Self::Value, De::Error>
    where
        De: Deserializer<'de>,
    {
        let model = MergeModel {
            storage: self.storage,
            refused: self.refused,
        };
        deserializer.deserialize_struct("Tokenizer", FIELDS, TokenizerVisitor::new(model))
    }
}

/// Reads one `model` value. `Failure` is a typed refusal outside serde errors.
trait ModelField<'de, M> {
    type Failure;

    fn next_model<A: MapAccess<'de>>(
        &self,
        map: &mut A,
    ) -> Result<Result<M, Self::Failure>, A::Error>;

    /// Keeps a failure that a later serde error would otherwise discard.
    fn keep(&self, failure: Self::Failure);
}

/// The original `Deserialize` route.
struct PlainModel;

impl<'de, M: Deserialize<'de>> ModelField<'de, M> for PlainModel {
    type Failure = Infallible;

    fn next_model<A: MapAccess<'de>>(
        &self,
        map: &mut A,
    ) -> Result<Result<M, Infallible>, A::Error> {
        map.next_value().map(Ok)
    }

    fn keep(&self, failure: Infallible) {
        match failure {}
    }
}

/// The merge storage route, with the caller's per-call refusal slot.
struct MergeModel<'a, S: MergeStorage> {
    storage: &'a S,
    refused: &'a Cell<Option<MergeBuildError<S::Error>>>,
}

impl<'de, S: MergeStorage> ModelField<'de, ModelWrapper> for MergeModel<'_, S> {
    type Failure = MergeBuildError<S::Error>;

    fn next_model<A: MapAccess<'de>>(
        &self,
        map: &mut A,
    ) -> Result<Result<ModelWrapper, Self::Failure>, A::Error> {
        map.next_value_seed(ModelMergeSeed {
            storage: self.storage,
        })
    }

    fn keep(&self, failure: Self::Failure) {
        self.refused.set(Some(failure));
    }
}

struct TokenizerVisitor<M, N, PT, PP, D, F>(
    F,
    PhantomData<M>,
    PhantomData<N>,
    PhantomData<PT>,
    PhantomData<PP>,
    PhantomData<D>,
);

impl<M, N, PT, PP, D, F> TokenizerVisitor<M, N, PT, PP, D, F> {
    fn new(model: F) -> Self {
        Self(
            model,
            PhantomData,
            PhantomData,
            PhantomData,
            PhantomData,
            PhantomData,
        )
    }
}

impl<'de, M, N, PT, PP, D, F> Visitor<'de> for TokenizerVisitor<M, N, PT, PP, D, F>
where
    M: Model,
    N: Deserialize<'de> + Normalizer,
    PT: Deserialize<'de> + PreTokenizer,
    PP: Deserialize<'de> + PostProcessor,
    D: Deserialize<'de> + Decoder,
    F: ModelField<'de, M>,
{
    type Value = Result<TokenizerImpl<M, N, PT, PP, D>, F::Failure>;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct Tokenizer")
    }

    fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut builder = TokenizerBuilder::new();
        let mut tokens: Vec<AddedTokenWithId> = vec![];
        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "version" => {
                    let v: String = map.next_value()?;
                    if &v != "1.0" {
                        return Err(Error::custom(format!("Unknown tokenizer version '{v}'")));
                    }
                }
                "truncation" => {
                    builder = builder.with_truncation(map.next_value()?);
                }
                "padding" => {
                    builder = builder.with_padding(map.next_value()?);
                }
                "added_tokens" => {
                    tokens = map.next_value()?;
                }
                "normalizer" => {
                    builder = builder.with_normalizer(map.next_value()?);
                }
                "pre_tokenizer" => {
                    builder = builder.with_pre_tokenizer(map.next_value()?);
                }
                "model" => match self.0.next_model(&mut map)? {
                    Ok(model) => builder = builder.with_model(model),
                    Err(failure) => {
                        // serde_json requires the closing brace after `visit_map`.
                        // Skip the remaining entries syntactically, constructing
                        // nothing; a skip error must not discard the failure.
                        return match skip_entries(&mut map) {
                            Ok(()) => Ok(Err(failure)),
                            Err(error) => {
                                self.0.keep(failure);
                                Err(error)
                            }
                        };
                    }
                },
                "decoder" => {
                    builder = builder.with_decoder(map.next_value()?);
                }
                "post_processor" => {
                    builder = builder.with_post_processor(map.next_value()?);
                }
                _ => {}
            };
        }
        let mut tokenizer = builder
            .build()
            .map_err(|e| V::Error::custom(e.to_string()))?;

        // Single-pass: warn on ID mismatches, then add all tokens.
        // `add_tokens` computes normalization internally for tokens with `normalized = true`.
        for t in &tokens {
            if let Some(rid) = tokenizer.token_to_id(&t.token.content) {
                if rid != t.id {
                    warn!(
                        "Warning: Token '{}' was expected to have ID '{}' but was given ID '{}'",
                        t.token.content, t.id, rid
                    );
                }
            }
        }
        tokenizer
            .add_tokens(tokens.into_iter().map(|t| t.token))
            .map_err(|e| V::Error::custom(e.to_string()))?;

        Ok(Ok(tokenizer))
    }
}

/// Skips every remaining entry with `IgnoredAny`. After `Ok`, the next byte is
/// the closing brace, so serde_json's map completion succeeds.
fn skip_entries<'de, A: MapAccess<'de>>(map: &mut A) -> Result<(), A::Error> {
    while map.next_key::<IgnoredAny>()?.is_some() {
        map.next_value::<IgnoredAny>()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::tokenizer::Tokenizer;
    use std::str::FromStr;

    #[test]
    fn test_deserialization_serialization_invariant() {
        let tok_json = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {
      "id": 0,
      "content": "[SPECIAL_0]",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 1,
      "content": "[SPECIAL_1]",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 2,
      "content": "[SPECIAL_2]",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "",
    "max_input_chars_per_word": 100,
    "vocab": {}
  }
}"#;
        let tokenizer = Tokenizer::from_str(tok_json).unwrap();

        let tok_str = serde_json::to_string_pretty(&tokenizer).unwrap();
        // It should be exactly the same as above
        assert_eq!(tok_str, tok_json);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_from_pretrained() {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_target(false)
            .init();
        let _ = Tokenizer::from_pretrained("Qwen/Qwen2-7B-Instruct", None);
        warn!("This should be the first warning");
    }
}
