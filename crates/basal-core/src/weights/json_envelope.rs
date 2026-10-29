//! Original-byte JSON preflight. Only the scanner itself uses no heap storage.

const MAX_DEPTH: usize = 64;
const MAX_NODES_AND_KEYS: u64 = 1_048_576;
const MAX_STRING_BYTES: u64 = 1024 * 1024;
const MAX_TOTAL_STRING_BYTES: u64 = 32 * 1024 * 1024;
const MAX_NUMBER_BYTES: u64 = 128;
const MAX_TOTAL_NUMBER_BYTES: u64 = 8 * 1024 * 1024;

/// Fixed per-document lexical resources on the opt-in JSON envelope route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JsonEnvelopeResource {
    Depth,
    NodesAndKeys,
    StringBytes,
    TotalStringBytes,
    NumberBytes,
    TotalNumberBytes,
}

/// Preflight refusal with a zero-based byte offset and no retained input text.
#[derive(Clone, Copy, Debug, thiserror::Error)]
pub enum JsonEnvelopeError {
    #[error("JSON envelope {resource:?} limit {limit} exceeded at byte {offset}")]
    Resource { resource: JsonEnvelopeResource, offset: usize, limit: u64 },
    #[error("malformed JSON at byte {offset}: {reason}")]
    Malformed { offset: usize, reason: &'static str },
    #[error("JSON envelope arithmetic overflow at byte {offset}")]
    Arithmetic { offset: usize },
}

/// Construction-only choice; existing public entry points always select Legacy.
#[derive(Clone, Copy)]
pub(super) enum JsonReadMode {
    Legacy,
    Envelope,
}

impl JsonReadMode {
    pub(super) fn check(self, bytes: &[u8]) -> Result<(), JsonEnvelopeError> {
        match self {
            Self::Legacy => Ok(()),
            Self::Envelope => check_json_envelope(bytes),
        }
    }
}

#[derive(Clone, Copy)]
enum Frame {
    ObjectFirstKey,
    ObjectKey,
    ObjectColon,
    ObjectValue,
    ObjectAfterValue,
    ArrayFirstValue,
    ArrayValue,
    ArrayAfterValue,
}

struct Scanner<'a> {
    bytes: &'a [u8],
    offset: usize,
    frames: [Frame; MAX_DEPTH],
    depth: usize,
    nodes_and_keys: u64,
    string_bytes: u64,
    number_bytes: u64,
}

/// Validate syntax and quotas without copies, recursion, serde or heap storage.
fn check_json_envelope(bytes: &[u8]) -> Result<(), JsonEnvelopeError> {
    std::str::from_utf8(bytes)
        .map_err(|error| JsonEnvelopeError::Malformed { offset: error.valid_up_to(), reason: "invalid UTF-8" })?;
    Scanner {
        bytes,
        offset: 0,
        frames: [Frame::ArrayFirstValue; MAX_DEPTH],
        depth: 0,
        nodes_and_keys: 0,
        string_bytes: 0,
        number_bytes: 0,
    }
    .document()
}

impl Scanner<'_> {
    fn malformed(&self, reason: &'static str) -> JsonEnvelopeError {
        JsonEnvelopeError::Malformed { offset: self.offset, reason }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.offset).copied()
    }

    fn advance(&mut self) -> Result<(), JsonEnvelopeError> {
        if self.peek().is_none() {
            return Err(self.malformed("unexpected end of input"));
        }
        self.offset = self.offset.checked_add(1).ok_or(JsonEnvelopeError::Arithmetic { offset: self.offset })?;
        Ok(())
    }

    fn whitespace(&mut self) -> Result<(), JsonEnvelopeError> {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.advance()?;
        }
        Ok(())
    }

    fn document(mut self) -> Result<(), JsonEnvelopeError> {
        let mut root_seen = false;
        loop {
            self.whitespace()?;
            if self.depth == 0 {
                if root_seen {
                    return if self.peek().is_none() { Ok(()) } else { Err(self.malformed("trailing input")) };
                }
                root_seen = true;
                self.value()?;
                continue;
            }
            // depth is positive here and only push grows it, after a capacity check.
            let top = self.depth - 1;
            match self.frames[top] {
                Frame::ObjectFirstKey if self.peek() == Some(b'}') => self.close()?,
                Frame::ObjectFirstKey | Frame::ObjectKey => {
                    if self.peek() != Some(b'"') {
                        return Err(self.malformed("expected object key"));
                    }
                    self.node()?;
                    self.string()?;
                    self.frames[top] = Frame::ObjectColon;
                }
                Frame::ObjectColon => {
                    self.required(b':', "expected colon")?;
                    self.frames[top] = Frame::ObjectValue;
                }
                Frame::ObjectValue => {
                    self.frames[top] = Frame::ObjectAfterValue;
                    self.value()?;
                }
                Frame::ArrayFirstValue if self.peek() == Some(b']') => self.close()?,
                Frame::ArrayFirstValue | Frame::ArrayValue => {
                    self.frames[top] = Frame::ArrayAfterValue;
                    self.value()?;
                }
                Frame::ObjectAfterValue | Frame::ArrayAfterValue => {
                    let object = matches!(self.frames[top], Frame::ObjectAfterValue);
                    if self.peek() == Some(if object { b'}' } else { b']' }) {
                        self.close()?;
                    } else {
                        self.required(b',', "expected comma or container end")?;
                        self.frames[top] = if object { Frame::ObjectKey } else { Frame::ArrayValue };
                    }
                }
            }
        }
    }

    fn node(&mut self) -> Result<(), JsonEnvelopeError> {
        charge(&mut self.nodes_and_keys, MAX_NODES_AND_KEYS, JsonEnvelopeResource::NodesAndKeys, self.offset)
    }

    fn value(&mut self) -> Result<(), JsonEnvelopeError> {
        let byte = self.peek().ok_or_else(|| self.malformed("expected JSON value"))?;
        self.node()?;
        match byte {
            b'{' => self.push(Frame::ObjectFirstKey),
            b'[' => self.push(Frame::ArrayFirstValue),
            b'"' => self.string(),
            b't' => self.literal(b"true"),
            b'f' => self.literal(b"false"),
            b'n' => self.literal(b"null"),
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(self.malformed("expected JSON value")),
        }
    }

    fn push(&mut self, frame: Frame) -> Result<(), JsonEnvelopeError> {
        if self.depth >= MAX_DEPTH {
            return Err(JsonEnvelopeError::Resource {
                resource: JsonEnvelopeResource::Depth,
                offset: self.offset,
                limit: 64,
            });
        }
        // Check above proves this index and depth+1 <= 64 before either write.
        self.frames[self.depth] = frame;
        self.depth += 1;
        self.advance()
    }

    fn close(&mut self) -> Result<(), JsonEnvelopeError> {
        // Called only for the matching delimiter of a frame with depth > 0.
        self.depth -= 1;
        self.advance()
    }

    fn required(&mut self, byte: u8, reason: &'static str) -> Result<(), JsonEnvelopeError> {
        if self.peek() != Some(byte) {
            return Err(self.malformed(reason));
        }
        self.advance()
    }

    fn literal(&mut self, literal: &[u8]) -> Result<(), JsonEnvelopeError> {
        for &byte in literal {
            self.required(byte, "invalid literal")?;
        }
        Ok(())
    }

    fn string_byte(&mut self, length: &mut u64) -> Result<u8, JsonEnvelopeError> {
        let byte = self.peek().ok_or_else(|| self.malformed("unterminated string"))?;
        charge(length, MAX_STRING_BYTES, JsonEnvelopeResource::StringBytes, self.offset)?;
        charge(&mut self.string_bytes, MAX_TOTAL_STRING_BYTES, JsonEnvelopeResource::TotalStringBytes, self.offset)?;
        self.advance()?;
        Ok(byte)
    }

    fn string(&mut self) -> Result<(), JsonEnvelopeError> {
        self.required(b'"', "expected string")?;
        let mut length = 0;
        loop {
            match self.peek() {
                Some(b'"') => return self.advance(),
                Some(0..=0x1f) => return Err(self.malformed("unescaped control byte")),
                Some(b'\\') => {
                    self.string_byte(&mut length)?;
                    match self.string_byte(&mut length)? {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {}
                        b'u' => {
                            let code = self.hex_quad(&mut length)?;
                            if (0xd800..=0xdbff).contains(&code) {
                                // The second escape is part of the same raw string body.
                                if self.string_byte(&mut length)? != b'\\' {
                                    return Err(self.malformed("expected low surrogate escape"));
                                }
                                if self.string_byte(&mut length)? != b'u' {
                                    return Err(self.malformed("expected low surrogate escape"));
                                }
                                let low = self.hex_quad(&mut length)?;
                                if !(0xdc00..=0xdfff).contains(&low) {
                                    return Err(self.malformed("invalid low surrogate"));
                                }
                            } else if (0xdc00..=0xdfff).contains(&code) {
                                return Err(self.malformed("unpaired low surrogate"));
                            }
                        }
                        _ => return Err(self.malformed("invalid string escape")),
                    }
                }
                Some(_) => {
                    self.string_byte(&mut length)?;
                }
                None => return Err(self.malformed("unterminated string")),
            }
        }
    }

    fn hex_quad(&mut self, length: &mut u64) -> Result<u16, JsonEnvelopeError> {
        let mut code = 0_u16;
        for _ in 0..4 {
            let nibble = match self.string_byte(length)? {
                byte @ b'0'..=b'9' => byte - b'0',
                byte @ b'a'..=b'f' => byte - b'a' + 10,
                byte @ b'A'..=b'F' => byte - b'A' + 10,
                _ => return Err(self.malformed("invalid Unicode escape")),
            };
            // Exactly four nibbles: before each shift code has at most 12 bits.
            code = (code << 4) | u16::from(nibble);
        }
        Ok(code)
    }

    fn number_byte(&mut self, length: &mut u64) -> Result<(), JsonEnvelopeError> {
        charge(length, MAX_NUMBER_BYTES, JsonEnvelopeResource::NumberBytes, self.offset)?;
        charge(&mut self.number_bytes, MAX_TOTAL_NUMBER_BYTES, JsonEnvelopeResource::TotalNumberBytes, self.offset)?;
        self.advance()
    }

    fn digits(&mut self, length: &mut u64) -> Result<(), JsonEnvelopeError> {
        if !matches!(self.peek(), Some(b'0'..=b'9')) {
            return Err(self.malformed("expected number digit"));
        }
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.number_byte(length)?;
        }
        Ok(())
    }

    fn number(&mut self) -> Result<(), JsonEnvelopeError> {
        let mut length = 0;
        if self.peek() == Some(b'-') {
            self.number_byte(&mut length)?;
        }
        match self.peek() {
            Some(b'0') => self.number_byte(&mut length)?,
            Some(b'1'..=b'9') => self.digits(&mut length)?,
            _ => return Err(self.malformed("expected number integer")),
        }
        if self.peek() == Some(b'.') {
            self.number_byte(&mut length)?;
            self.digits(&mut length)?;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.number_byte(&mut length)?;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.number_byte(&mut length)?;
            }
            self.digits(&mut length)?;
        }
        Ok(())
    }
}

fn charge(
    counter: &mut u64,
    limit: u64,
    resource: JsonEnvelopeResource,
    offset: usize,
) -> Result<(), JsonEnvelopeError> {
    let next = counter.checked_add(1).ok_or(JsonEnvelopeError::Arithmetic { offset })?;
    if next > limit {
        return Err(JsonEnvelopeError::Resource { resource, offset, limit });
    }
    *counter = next;
    Ok(())
}
