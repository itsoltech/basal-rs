//! Closed declaration schema planning and exact, held semantic backing.
//!
//! Envelope runs first. This cursor still checks every offset, shape and scalar;
//! it never deserializes, skips an arbitrary subtree or builds a temporary heap.

use std::alloc::{alloc, Layout};

use super::raw_source::RawCharge;
use super::{
    logical_origin, GgufDeclaration, GgufOuttype, GgufRawSourceBudget, GgufRawSourceError, MetadataRepair,
    OriginSafetensor, Sha256Digest, WeightSourceError, CONVERTER_COMMIT, EXPORTER_COMMIT,
};

/// Shared allowance for original raw content and final declaration heap layouts.
///
/// Create one before loads and share it (or its clones) across concurrent loads
/// and reloads. Other semantic/native/tokenizer/cache heaps, containing objects,
/// inventory names/records, paths/errors, GGUF scratch and device memory remain
/// unadmitted. This is not a bound on allocator internals, RSS or physical OOM.
#[derive(Clone)]
pub struct GgufDeclarationSourceBudget {
    raw: GgufRawSourceBudget,
}

impl GgufDeclarationSourceBudget {
    /// Create one independent allowance; equality with the limit is accepted.
    /// Bootstrap/control allocation remains infallible and outside accounting.
    pub fn new(limit_bytes: usize) -> Self {
        Self { raw: GgufRawSourceBudget::new(limit_bytes) }
    }

    pub(super) fn raw(&self) -> &GgufRawSourceBudget {
        &self.raw
    }
}

/// Fixed construction causes for the new declaration-budget route only.
/// No variant retains input text, paths, buffers or an allocation permit.
#[derive(Debug, thiserror::Error)]
pub enum GgufDeclarationConstructionError {
    #[error("declaration field {field} at byte {offset}: {reason}")]
    Schema { offset: usize, field: &'static str, reason: &'static str },
    #[error("declaration construction arithmetic overflows")]
    Arithmetic,
    #[error("invalid declaration {storage} layout for {count} elements")]
    Layout { storage: &'static str, count: usize },
    #[error("declaration source admission failed")]
    Admission {
        #[source]
        source: GgufRawSourceError,
    },
    #[error("declaration {storage} allocation of {bytes} bytes failed")]
    Allocation { storage: &'static str, bytes: usize },
}

use GgufDeclarationConstructionError as Error;

/// A closed construction choice: the declaration variant supplies BOTH raw
/// reads and semantic admission from its one hidden ledger. No upgrade exists.
#[derive(Clone, Copy)]
pub(super) enum SourceBudget<'a> {
    None,
    Raw(&'a GgufRawSourceBudget),
    Declaration(&'a GgufDeclarationSourceBudget),
}

impl<'a> SourceBudget<'a> {
    pub(super) fn raw(self) -> Option<&'a GgufRawSourceBudget> {
        match self {
            Self::None => None,
            Self::Raw(raw) => Some(raw),
            Self::Declaration(budget) => Some(&budget.raw),
        }
    }

    pub(super) fn declaration(self, bytes: &[u8]) -> Result<GgufDeclaration, WeightSourceError> {
        match self {
            Self::None | Self::Raw(_) => GgufDeclaration::parse(bytes),
            Self::Declaration(budget) => DeclarationPlan::scan(bytes)
                .and_then(|plan| plan.materialize(budget))
                .map_err(|source| WeightSourceError::DeclarationConstruction { source }),
        }
    }
}

/// Non-Clone, private, last field of the declaration. Its raw charge uses the
/// existing ledger arithmetic and cleanup; it does not own another counter.
pub(super) struct DeclarationCharge {
    _raw: RawCharge,
}

#[derive(Clone, Copy)]
struct Text<'a> {
    source: &'a str,
    start: usize,
    end: usize,
    bytes: usize,
    field: &'static str,
}

impl<'a> Text<'a> {
    fn cursor(self) -> Cursor<'a> {
        Cursor { source: self.source, offset: self.start, end: self.end, field: self.field }
    }

    fn equals(self, expected: &str) -> Result<bool, Error> {
        let mut cursor = self.cursor();
        let mut expected = expected.chars();
        while cursor.peek().is_some() {
            if Some(cursor.scalar()?) != expected.next() {
                return Ok(false);
            }
        }
        Ok(expected.next().is_none())
    }

    fn exact(self, expected: &str) -> Result<(), Error> {
        if !self.equals(expected)? {
            return Err(self.cursor().error("unsupported constant"));
        }
        Ok(())
    }

    fn layout(self) -> Result<Layout, Error> {
        Layout::array::<u8>(self.bytes).map_err(|_| Error::Layout { storage: self.field, count: self.bytes })
    }

    fn digest(self) -> Result<Sha256Digest, Error> {
        if self.bytes != 64 {
            return Err(self.cursor().error("expected 64 ASCII hex characters"));
        }
        let mut cursor = self.cursor();
        let mut digest = [0; 32];
        for byte in &mut digest {
            *byte = (cursor.hex_scalar()? << 4) | cursor.hex_scalar()?;
        }
        if cursor.peek().is_some() {
            return Err(cursor.error("expected 64 ASCII hex characters"));
        }
        Ok(Sha256Digest(digest))
    }

    /// Caller holds the aggregate charge, including this name before insertion.
    fn materialize(self) -> Result<String, Error> {
        let layout = self.layout()?;
        let mut text = if self.bytes == 0 {
            String::new()
        } else {
            // SAFETY: array<u8> validates nonzero size/isize bound and alignment
            // 1. The caller already holds S. Global alloc matches String's
            // deallocator; no lock or borrowed heap storage is involved. Null
            // is checked before adoption, without dereferencing the pointer.
            let pointer = unsafe { alloc(layout) };
            if pointer.is_null() {
                return Err(Error::Allocation { storage: self.field, bytes: layout.size() });
            }
            // SAFETY: Unique immediate adoption, with no intervening fallible
            // work. Length 0 is valid UTF-8 and exposes no uninitialized bytes;
            // capacity bytes/alignment 1 exactly reproduce the global layout.
            // Only bounded scalar pushes below initialize bytes, never reserve.
            // No raw pointer escapes. String owns deallocation and supplies its
            // usual Send/Sync; the caller's charge outlives errors/unwind/moves.
            unsafe { String::from_raw_parts(pointer, 0, self.bytes) }
        };
        let mut cursor = self.cursor();
        while cursor.peek().is_some() {
            let scalar = cursor.scalar()?;
            let length = add(text.len(), scalar.len_utf8())?;
            if length > self.bytes || length > text.capacity() {
                return Err(cursor.error("decoded text exceeds planned capacity"));
            }
            text.push(scalar);
        }
        if text.len() != self.bytes {
            return Err(cursor.error("decoded text differs from plan"));
        }
        Ok(text)
    }
}

struct RecordsPlan<'a> {
    source: &'a str,
    start: usize,
    end: usize,
    count: usize,
    name_bytes: usize,
}

impl RecordsPlan<'_> {
    fn layout(&self) -> Result<Layout, Error> {
        Layout::array::<OriginSafetensor>(self.count)
            .map_err(|_| Error::Layout { storage: "origin.safetensors", count: self.count })
    }

    /// Closed typed allocation: never allocate byte-aligned storage for records.
    fn materialize(&self, layout: Layout) -> Result<Vec<OriginSafetensor>, Error> {
        let mut records = if layout.size() == 0 {
            Vec::new()
        } else {
            // SAFETY: layout is exactly array<OriginSafetensor>(count), checked
            // before admission, with target record alignment and isize bound.
            // The caller holds the whole charge. Global alloc matches Vec's
            // allocator. No lock is held; null is checked before adoption.
            let pointer = unsafe { alloc(layout) }.cast::<OriginSafetensor>();
            if pointer.is_null() {
                return Err(Error::Allocation { storage: "origin.safetensors", bytes: layout.size() });
            }
            // SAFETY: Immediately and uniquely adopt the global allocation at
            // its actual typed alignment/capacity. Length 0 exposes no records.
            // Bounded pushes below initialize whole records; Vec drops exactly
            // that initialized prefix, including their names, then deallocates
            // with this same layout. No pointer/owned record escapes; the caller
            // holds its earlier charge through partial failure/unwind and moves.
            unsafe { Vec::from_raw_parts(pointer, 0, self.count) }
        };
        let mut cursor = Cursor { source: self.source, offset: self.start, end: self.end, field: "origin.safetensors" };
        cursor.required(b'[')?;
        while cursor.array_item(records.is_empty())? {
            // Refuse mismatches before constructing another name or growing Vec.
            if records.len() >= self.count || records.len() >= records.capacity() {
                return Err(cursor.error("record count exceeds planned capacity"));
            }
            let (span, sha256) = cursor.record()?;
            let name = span.materialize()?;
            if !name.ends_with(".safetensors")
                || name.contains(['/', '\\', ':'])
                || name.chars().any(char::is_control)
                || name == "."
                || name == ".."
            {
                return Err(span.cursor().error("expected one logical .safetensors filename"));
            }
            if records.iter().any(|previous: &OriginSafetensor| previous.name == name) {
                return Err(span.cursor().error("duplicate filename"));
            }
            records.push(OriginSafetensor { name, sha256 });
        }
        if records.len() != self.count || cursor.offset != self.end {
            return Err(cursor.error("record array differs from plan"));
        }
        Ok(records)
    }
}

struct OriginPlan<'a> {
    model: Text<'a>,
    revision: Text<'a>,
    records: RecordsPlan<'a>,
}

struct DeclarationPlan<'a> {
    origin: OriginPlan<'a>,
    gguf_sha256: Sha256Digest,
    companion_sha256: [Sha256Digest; 5],
    outtype: GgufOuttype,
    repair: Option<MetadataRepair>,
}

impl<'a> DeclarationPlan<'a> {
    fn scan(bytes: &'a [u8]) -> Result<Self, Error> {
        let source = std::str::from_utf8(bytes).map_err(|e| Error::Schema {
            offset: e.valid_up_to(),
            field: "declaration",
            reason: "invalid UTF-8",
        })?;
        let mut cursor = Cursor { source, offset: 0, end: source.len(), field: "declaration" };
        cursor.required(b'{')?;
        let mut seen = 0;
        let mut origin = None;
        let mut gguf_sha256 = Sha256Digest([0; 32]);
        let mut companion_sha256 = [Sha256Digest([0; 32]); 5];
        let mut outtype = GgufOuttype::F16;
        let mut repair = None;
        while let Some(field) = cursor.field(
            &["schema", "layout", "gguf_sha256", "converter", "companion_sha256", "origin", "metadata_repair"],
            &mut seen,
        )? {
            match field {
                "schema" => cursor.text(field)?.exact("basal.gguf-source.1")?,
                "layout" => cursor.text(field)?.exact("llama-cpp-hf-llama-v1")?,
                "gguf_sha256" => gguf_sha256 = cursor.text(field)?.digest()?,
                "converter" => outtype = cursor.converter()?,
                "companion_sha256" => companion_sha256 = cursor.companions()?,
                "origin" => origin = Some(cursor.origin()?),
                "metadata_repair" => repair = cursor.repair()?,
                _ => return Err(cursor.error("unknown field")),
            }
        }
        cursor.whitespace()?;
        if cursor.peek().is_some() {
            return Err(cursor.error("trailing input"));
        }
        Ok(Self { origin: cursor.present(origin)?, gguf_sha256, companion_sha256, outtype, repair })
    }

    fn materialize(self, budget: &GgufDeclarationSourceBudget) -> Result<GgufDeclaration, Error> {
        let records_layout = self.origin.records.layout()?;
        self.origin.model.layout()?;
        self.origin.revision.layout()?;
        // Every name layout was checked during Pass A, including duplicates.
        let strings = add(add(self.origin.model.bytes, self.origin.revision.bytes)?, self.origin.records.name_bytes)?;
        let requested = add(records_layout.size(), strings)?;
        // Charge local precedes ALL selected heap locals. Reverse local drop
        // order keeps S held through null/partial/schema/validation/unwind paths.
        let mut raw = budget.raw.charge();
        raw.acquire(requested).map_err(|source| Error::Admission { source })?;
        let charge = DeclarationCharge { _raw: raw };
        let model_id = self.origin.model.materialize()?;
        check_origin(&model_id, self.origin.model)?;
        let revision = self.origin.revision.materialize()?;
        check_origin(&revision, self.origin.revision)?;
        let safetensors = self.origin.records.materialize(records_layout)?;
        Ok(GgufDeclaration {
            gguf_sha256: self.gguf_sha256,
            companion_sha256: self.companion_sha256,
            outtype: self.outtype,
            model_id,
            revision,
            safetensors,
            metadata_repair: self.repair,
            _declaration_charge: Some(charge),
        })
    }
}

fn check_origin(value: &str, span: Text<'_>) -> Result<(), Error> {
    // The existing predicate returns only static fields/reasons and allocates
    // nothing. Its old-route diagnostic remains unchanged.
    logical_origin(value, span.field)
        .map_err(|_| span.cursor().error("expected a nonempty logical origin, without controls, absolute paths or URI"))
}

fn add(left: usize, right: usize) -> Result<usize, Error> {
    left.checked_add(right).ok_or(Error::Arithmetic)
}

struct Cursor<'a> {
    source: &'a str,
    offset: usize,
    end: usize,
    field: &'static str,
}

impl<'a> Cursor<'a> {
    fn error(&self, reason: &'static str) -> Error {
        Error::Schema { offset: self.offset, field: self.field, reason }
    }

    fn peek(&self) -> Option<u8> {
        if self.offset < self.end {
            self.source.as_bytes().get(self.offset).copied()
        } else {
            None
        }
    }

    fn advance(&mut self, count: usize) -> Result<(), Error> {
        let end = add(self.offset, count)?;
        if end > self.end || end > self.source.len() {
            return Err(self.error("unexpected end of input"));
        }
        self.offset = end;
        Ok(())
    }

    fn byte(&mut self) -> Result<u8, Error> {
        let byte = self.peek().ok_or_else(|| self.error("unexpected end of input"))?;
        self.advance(1)?;
        Ok(byte)
    }

    fn whitespace(&mut self) -> Result<(), Error> {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.advance(1)?;
        }
        Ok(())
    }

    fn required(&mut self, byte: u8) -> Result<(), Error> {
        self.whitespace()?;
        if self.peek() != Some(byte) {
            return Err(self.error("unexpected schema token"));
        }
        self.advance(1)
    }

    /// Closed schema fields only; no callback, recursive skip or decoded key heap.
    /// A separate seen bit enforces required-null and escaped duplicate equality.
    fn field(&mut self, keys: &[&'static str], seen: &mut u16) -> Result<Option<&'static str>, Error> {
        self.whitespace()?;
        if self.peek() == Some(b'}') {
            let complete = 1_u16
                .checked_shl(u32::try_from(keys.len()).map_err(|_| Error::Arithmetic)?)
                .and_then(|mask| mask.checked_sub(1))
                .ok_or(Error::Arithmetic)?;
            if *seen != complete {
                return Err(self.error("missing required field"));
            }
            self.advance(1)?;
            return Ok(None);
        }
        if *seen != 0 {
            self.required(b',')?;
        }
        let key = self.text(self.field)?;
        for (index, &field) in keys.iter().enumerate() {
            if key.equals(field)? {
                let bit =
                    1_u16.checked_shl(u32::try_from(index).map_err(|_| Error::Arithmetic)?).ok_or(Error::Arithmetic)?;
                if *seen & bit != 0 {
                    return Err(key.cursor().error("duplicate field"));
                }
                *seen |= bit;
                self.required(b':')?;
                return Ok(Some(field));
            }
        }
        Err(key.cursor().error("unknown field"))
    }

    fn present<T>(&self, value: Option<T>) -> Result<T, Error> {
        value.ok_or_else(|| self.error("missing required field"))
    }

    fn text(&mut self, field: &'static str) -> Result<Text<'a>, Error> {
        self.field = field;
        self.required(b'"')?;
        let start = self.offset;
        let mut bytes = 0;
        while self.peek() != Some(b'"') {
            bytes = add(bytes, self.scalar()?.len_utf8())?;
        }
        let end = self.offset;
        self.advance(1)?;
        Ok(Text { source: self.source, start, end, bytes, field })
    }

    fn scalar(&mut self) -> Result<char, Error> {
        match self.peek() {
            Some(b'\\') => {
                self.advance(1)?;
                match self.byte()? {
                    b'"' => Ok('"'),
                    b'\\' => Ok('\\'),
                    b'/' => Ok('/'),
                    b'b' => Ok('\u{0008}'),
                    b'f' => Ok('\u{000c}'),
                    b'n' => Ok('\n'),
                    b'r' => Ok('\r'),
                    b't' => Ok('\t'),
                    b'u' => {
                        let high = self.hex_quad()?;
                        let code = if (0xd800..=0xdbff).contains(&high) {
                            // No whitespace is legal between surrogate escapes.
                            if self.byte()? != b'\\' || self.byte()? != b'u' {
                                return Err(self.error("expected low surrogate escape"));
                            }
                            let low = self.hex_quad()?;
                            if !(0xdc00..=0xdfff).contains(&low) {
                                return Err(self.error("invalid low surrogate"));
                            }
                            // Both operands are bounded 10-bit values.
                            0x10000 + ((high - 0xd800) << 10) + (low - 0xdc00)
                        } else {
                            high
                        };
                        char::from_u32(code).ok_or_else(|| self.error("invalid Unicode scalar"))
                    }
                    _ => Err(self.error("invalid string escape")),
                }
            }
            Some(0..=0x1f | b'"') => Err(self.error("invalid unescaped string scalar")),
            Some(_) => {
                let tail =
                    self.source.get(self.offset..self.end).ok_or_else(|| self.error("invalid UTF-8 boundary"))?;
                let scalar = tail.chars().next().ok_or_else(|| self.error("expected string scalar"))?;
                self.advance(scalar.len_utf8())?;
                Ok(scalar)
            }
            None => Err(self.error("unterminated string")),
        }
    }

    fn hex_quad(&mut self) -> Result<u32, Error> {
        let mut code = 0_u32;
        for _ in 0..4 {
            let nibble = match self.byte()? {
                byte @ b'0'..=b'9' => byte - b'0',
                byte @ b'a'..=b'f' => byte - b'a' + 10,
                byte @ b'A'..=b'F' => byte - b'A' + 10,
                _ => return Err(self.error("invalid Unicode escape")),
            };
            code = (code << 4) | u32::from(nibble);
        }
        Ok(code)
    }

    fn hex_scalar(&mut self) -> Result<u8, Error> {
        match self.scalar()? {
            scalar @ '0'..='9' => Ok(scalar as u8 - b'0'),
            scalar @ 'a'..='f' => Ok(scalar as u8 - b'a' + 10),
            scalar @ 'A'..='F' => Ok(scalar as u8 - b'A' + 10),
            _ => Err(self.error("expected ASCII hex character")),
        }
    }

    fn converter(&mut self) -> Result<GgufOuttype, Error> {
        self.field = "converter";
        self.required(b'{')?;
        let mut seen = 0;
        let mut outtype = GgufOuttype::F16;
        while let Some(field) = self.field(
            &[
                "repository",
                "commit",
                "class",
                "architecture",
                "outtype",
                "endianness",
                "split",
                "tensor_overrides",
                "weight_postprocessing",
            ],
            &mut seen,
        )? {
            match field {
                "repository" => self.text(field)?.exact("ggml-org/llama.cpp")?,
                "commit" => self.text(field)?.exact(CONVERTER_COMMIT)?,
                "class" => self.text(field)?.exact("conversion.llama.LlamaModel")?,
                "architecture" => self.text(field)?.exact("LlamaForCausalLM")?,
                "endianness" => self.text(field)?.exact("little")?,
                "outtype" => {
                    let text = self.text(field)?;
                    outtype = if text.equals("f16")? {
                        GgufOuttype::F16
                    } else if text.equals("bf16")? {
                        GgufOuttype::Bf16
                    } else if text.equals("f32")? {
                        GgufOuttype::F32
                    } else {
                        return Err(text.cursor().error("expected f16, bf16 or f32"));
                    };
                }
                "split" | "tensor_overrides" | "weight_postprocessing" => {
                    self.field = field;
                    self.literal(b"false")?;
                }
                _ => return Err(self.error("unknown field")),
            }
        }
        Ok(outtype)
    }

    fn companions(&mut self) -> Result<[Sha256Digest; 5], Error> {
        self.field = "companion_sha256";
        self.required(b'{')?;
        let mut seen = 0;
        let mut hashes = [Sha256Digest([0; 32]); 5];
        const KEYS: [&str; 5] =
            ["config.json", "basal.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"];
        while let Some(field) = self.field(&KEYS, &mut seen)? {
            let index = KEYS.iter().position(|&key| key == field).ok_or_else(|| self.error("unknown field"))?;
            let digest = self.text(field)?.digest()?;
            *hashes.get_mut(index).ok_or_else(|| self.error("invalid companion slot"))? = digest;
        }
        Ok(hashes)
    }

    fn origin(&mut self) -> Result<OriginPlan<'a>, Error> {
        self.field = "origin";
        self.required(b'{')?;
        let mut seen = 0;
        let mut model = None;
        let mut revision = None;
        let mut records = None;
        while let Some(field) = self.field(&["model_id", "revision", "safetensors"], &mut seen)? {
            match field {
                "model_id" => model = Some(self.text("origin.model_id")?),
                "revision" => revision = Some(self.text("origin.revision")?),
                "safetensors" => records = Some(self.records()?),
                _ => return Err(self.error("unknown field")),
            }
        }
        Ok(OriginPlan {
            model: self.present(model)?,
            revision: self.present(revision)?,
            records: self.present(records)?,
        })
    }

    fn array_item(&mut self, first: bool) -> Result<bool, Error> {
        self.whitespace()?;
        if self.peek() == Some(b']') {
            self.advance(1)?;
            return Ok(false);
        }
        if !first {
            self.required(b',')?;
        }
        Ok(true)
    }

    fn records(&mut self) -> Result<RecordsPlan<'a>, Error> {
        self.field = "origin.safetensors";
        self.whitespace()?;
        let start = self.offset;
        self.required(b'[')?;
        let mut count = 0;
        let mut name_bytes = 0;
        while self.array_item(count == 0)? {
            let (name, _) = self.record()?;
            name.layout()?;
            count = add(count, 1)?;
            name_bytes = add(name_bytes, name.bytes)?;
        }
        if count == 0 {
            return Err(self.error("origin.safetensors must be nonempty"));
        }
        Ok(RecordsPlan { source: self.source, start, end: self.offset, count, name_bytes })
    }

    fn record(&mut self) -> Result<(Text<'a>, Sha256Digest), Error> {
        self.field = "origin.safetensors";
        self.required(b'{')?;
        let mut seen = 0;
        let mut name = None;
        let mut sha256 = Sha256Digest([0; 32]);
        while let Some(field) = self.field(&["name", "sha256"], &mut seen)? {
            match field {
                "name" => name = Some(self.text("origin.safetensors.name")?),
                "sha256" => sha256 = self.text("origin.safetensors.sha256")?.digest()?,
                _ => return Err(self.error("unknown field")),
            }
        }
        Ok((self.present(name)?, sha256))
    }

    fn literal(&mut self, literal: &[u8]) -> Result<(), Error> {
        self.whitespace()?;
        for &byte in literal {
            if self.byte()? != byte {
                return Err(self.error("unexpected schema literal"));
            }
        }
        Ok(())
    }

    fn repair(&mut self) -> Result<Option<MetadataRepair>, Error> {
        self.field = "metadata_repair";
        self.whitespace()?;
        if self.peek() == Some(b'n') {
            self.literal(b"null")?;
            return Ok(None);
        }
        self.required(b'{')?;
        let mut seen = 0;
        while let Some(field) = self.field(&["kind", "basal_exporter_commit", "gguf_copy_commit"], &mut seen)? {
            match field {
                "kind" => self.text(field)?.exact("tokenizer-only")?,
                "basal_exporter_commit" => self.text(field)?.exact(EXPORTER_COMMIT)?,
                "gguf_copy_commit" => self.text(field)?.exact(CONVERTER_COMMIT)?,
                _ => return Err(self.error("unknown field")),
            }
        }
        Ok(Some(MetadataRepair::TokenizerOnly))
    }
}
