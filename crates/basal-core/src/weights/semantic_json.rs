//! Flat immutable JSON backing. Both passes use the same fixed-stack cursor.

use std::alloc::{alloc, Layout};

use super::raw_source::RawCharge;
use super::{GgufDeclarationSourceBudget, GgufRawSourceError, Sha256Digest};

/// Fixed construction causes; no input text, backing or permit escapes in errors.
#[derive(Debug, thiserror::Error)]
pub enum SemanticConstructionError {
    #[error("semantic JSON syntax at byte {offset}: {reason}")]
    Syntax { offset: usize, reason: &'static str },
    #[error("semantic JSON number at byte {offset}: {reason}")]
    Number { offset: usize, reason: &'static str },
    #[error("semantic JSON construction arithmetic overflows")]
    Arithmetic,
    #[error("invalid semantic {storage} layout for {count} elements")]
    Layout { storage: &'static str, count: usize },
    #[error("semantic source admission failed")]
    Admission {
        #[source]
        source: GgufRawSourceError,
    },
    #[error("semantic {storage} allocation of {bytes} bytes failed")]
    Allocation { storage: &'static str, bytes: usize },
    #[error("semantic {storage} differs from plan at byte {offset}")]
    PlanMismatch { storage: &'static str, offset: usize },
}

use SemanticConstructionError as Error;

#[derive(Clone, Copy)]
pub(super) struct NodeId(usize);
#[derive(Clone, Copy)]
struct EdgeId(usize);
#[derive(Clone, Copy)]
pub(super) struct TextSpan {
    start: usize,
    len: usize,
}
#[derive(Clone, Copy)]
struct NumberSpan {
    start: usize,
    len: usize,
}
#[derive(Clone, Copy)]
struct Node {
    source_offset: usize,
    value: NodeValue,
}
#[derive(Clone, Copy)]
enum NodeValue {
    Null,
    Bool(bool),
    Number(NumberSpan),
    String(TextSpan),
    Array { first: Option<EdgeId>, len: usize },
    Object { first: Option<EdgeId>, len: usize },
}
#[derive(Clone, Copy)]
struct Edge {
    next: Option<EdgeId>,
    key: Option<TextSpan>,
    key_offset: Option<usize>,
    value: NodeId,
}

pub(super) struct SemanticDocument {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    text: Vec<u8>,
    numbers: Vec<u8>,
    root: NodeId,
    source_sha256: Sha256Digest,
    // Last: every covered allocation is destroyed before its charge.
    _charge: RawCharge,
}

impl SemanticDocument {
    pub(super) fn root(&self) -> ValueRef<'_> {
        self.value(self.root)
    }
    pub(super) fn value(&self, id: NodeId) -> ValueRef<'_> {
        ValueRef { doc: self, id }
    }
    pub(super) fn text(&self, span: TextSpan) -> &str {
        span_text(&self.text, span)
    }
    pub(super) fn sha256(&self) -> Sha256Digest {
        self.source_sha256
    }
}

/// JSON kind without an owned payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufSemanticKind {
    Null,
    Bool,
    Number,
    String,
    Array,
    Object,
}

/// Borrowed arbitrary immutable subtree. Copies of this view retain the owner borrow.
#[derive(Clone, Copy)]
pub struct GgufSemanticValue<'a> {
    doc: &'a SemanticDocument,
    id: NodeId,
}
pub(super) type ValueRef<'a> = GgufSemanticValue<'a>;

impl<'a> GgufSemanticValue<'a> {
    fn node(self) -> &'a Node {
        // IDs are private: the cursor inserts a complete placeholder before
        // linking it, and only that same document can create a borrowed view.
        &self.doc.nodes[self.id.0]
    }
    pub(super) fn id(self) -> NodeId {
        self.id
    }
    pub(super) fn offset(self) -> usize {
        self.node().source_offset
    }
    pub(super) fn text_span(self) -> Option<TextSpan> {
        match self.node().value {
            NodeValue::String(span) => Some(span),
            _ => None,
        }
    }
    pub fn kind(self) -> GgufSemanticKind {
        match self.node().value {
            NodeValue::Null => GgufSemanticKind::Null,
            NodeValue::Bool(_) => GgufSemanticKind::Bool,
            NodeValue::Number(_) => GgufSemanticKind::Number,
            NodeValue::String(_) => GgufSemanticKind::String,
            NodeValue::Array { .. } => GgufSemanticKind::Array,
            NodeValue::Object { .. } => GgufSemanticKind::Object,
        }
    }
    pub fn is_null(self) -> bool {
        self.kind() == GgufSemanticKind::Null
    }
    pub fn as_bool(self) -> Option<bool> {
        match self.node().value {
            NodeValue::Bool(value) => Some(value),
            _ => None,
        }
    }
    pub fn as_str(self) -> Option<&'a str> {
        self.text_span().map(|span| self.doc.text(span))
    }
    pub fn number_text(self) -> Option<&'a str> {
        match self.node().value {
            NodeValue::Number(span) => {
                std::str::from_utf8(self.doc.numbers.get(span.start..span.start.checked_add(span.len)?)?).ok()
            }
            _ => None,
        }
    }
    pub fn as_u64(self) -> Option<u64> {
        self.number_text()?.parse().ok()
    }
    pub fn as_i64(self) -> Option<i64> {
        self.number_text()?.parse().ok()
    }
    pub fn as_f64(self) -> Option<f64> {
        self.number_text()?.parse().ok().filter(|value: &f64| value.is_finite())
    }
    pub fn object(self) -> Option<GgufSemanticObject<'a>> {
        match self.node().value {
            NodeValue::Object { first, len } => Some(GgufSemanticObject { doc: Some(self.doc), first, len }),
            _ => None,
        }
    }
    pub fn array(self) -> Option<GgufSemanticArray<'a>> {
        match self.node().value {
            NodeValue::Array { first, len } => Some(GgufSemanticArray { doc: self.doc, first, len }),
            _ => None,
        }
    }
    pub fn get(self, key: &str) -> Option<Self> {
        self.object()?.get(key)
    }
    pub fn at(self, index: usize) -> Option<Self> {
        self.array()?.at(index)
    }
}

/// Insertion-ordered borrowed object, including a virtual empty calibration object.
#[derive(Clone, Copy)]
pub struct GgufSemanticObject<'a> {
    doc: Option<&'a SemanticDocument>,
    first: Option<EdgeId>,
    len: usize,
}
impl<'a> GgufSemanticObject<'a> {
    pub(super) fn empty() -> Self {
        Self { doc: None, first: None, len: 0 }
    }
    pub fn len(self) -> usize {
        self.len
    }
    pub fn is_empty(self) -> bool {
        self.len == 0
    }
    pub fn get(self, key: &str) -> Option<ValueRef<'a>> {
        self.members().find(|(name, _)| *name == key).map(|(_, value)| value)
    }
    pub fn members(self) -> GgufSemanticMembers<'a> {
        GgufSemanticMembers { doc: self.doc, next: self.first, remaining: self.len }
    }
}
/// Bounded iterator over visible members; duplicate keys keep their first position.
#[derive(Clone)]
pub struct GgufSemanticMembers<'a> {
    doc: Option<&'a SemanticDocument>,
    next: Option<EdgeId>,
    remaining: usize,
}
impl<'a> GgufSemanticMembers<'a> {
    pub(super) fn next_with_offset(&mut self) -> Option<(&'a str, ValueRef<'a>, usize)> {
        if self.remaining == 0 {
            return None;
        }
        let doc = self.doc?;
        let edge = doc.edges.get(self.next?.0)?;
        self.next = edge.next;
        self.remaining -= 1;
        Some((doc.text(edge.key?), doc.value(edge.value), edge.key_offset?))
    }
}
impl<'a> Iterator for GgufSemanticMembers<'a> {
    type Item = (&'a str, GgufSemanticValue<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let doc = self.doc?;
        let edge = doc.edges.get(self.next?.0)?;
        self.next = edge.next;
        self.remaining -= 1;
        Some((doc.text(edge.key?), doc.value(edge.value)))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
impl ExactSizeIterator for GgufSemanticMembers<'_> {}

/// Borrowed array with bounded traversal, without recursive owned nodes.
#[derive(Clone, Copy)]
pub struct GgufSemanticArray<'a> {
    doc: &'a SemanticDocument,
    first: Option<EdgeId>,
    len: usize,
}
impl<'a> GgufSemanticArray<'a> {
    pub fn len(self) -> usize {
        self.len
    }
    pub fn is_empty(self) -> bool {
        self.len == 0
    }
    pub fn at(self, index: usize) -> Option<ValueRef<'a>> {
        self.elements().nth(index)
    }
    pub fn elements(self) -> GgufSemanticElements<'a> {
        GgufSemanticElements { doc: self.doc, next: self.first, remaining: self.len }
    }
}
/// Bounded iterator over array elements in source order.
#[derive(Clone)]
pub struct GgufSemanticElements<'a> {
    doc: &'a SemanticDocument,
    next: Option<EdgeId>,
    remaining: usize,
}
impl<'a> Iterator for GgufSemanticElements<'a> {
    type Item = GgufSemanticValue<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let edge = self.doc.edges.get(self.next?.0)?;
        self.next = edge.next;
        self.remaining -= 1;
        Some(self.doc.value(edge.value))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
impl ExactSizeIterator for GgufSemanticElements<'_> {}

fn span_text(bytes: &[u8], span: TextSpan) -> &str {
    // Only the checked cursor creates spans, from complete encoded UTF-8 scalars.
    // Every end is a prefix of the fixed text buffer, whose final count is
    // checked before publication. Safe access also avoids unsafe UTF-8/lifetimes.
    span.start
        .checked_add(span.len)
        .and_then(|end| bytes.get(span.start..end))
        .and_then(|bytes| std::str::from_utf8(bytes).ok())
        .unwrap_or("")
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Arithmetic)
}
fn layout<T>(count: usize, storage: &'static str) -> Result<Layout, Error> {
    Layout::array::<T>(count).map_err(|_| Error::Layout { storage, count })
}

/// Private typed allocation, called only for Node, Edge and u8 after aggregate admission.
fn allocate_exact<T: Copy>(count: usize, layout: Layout, storage: &'static str) -> Result<Vec<T>, Error> {
    if count == 0 {
        return Ok(Vec::new());
    }
    // SAFETY: Each private call passes exactly Layout::array::<T>(count), checked
    // before acquire; the concrete types have nonzero size. Its size/alignment
    // match Vec<T>'s global allocator. The caller's earlier charge outlives all
    // allocated locals. No lock is held. Null is checked before adoption.
    let pointer = unsafe { alloc(layout) }.cast::<T>();
    if pointer.is_null() {
        return Err(Error::Allocation { storage, bytes: layout.size() });
    }
    // SAFETY: Immediate unique adoption at the exact typed alignment/capacity,
    // length zero. Only bounded pushes initialize elements. Vec drops that prefix
    // and deallocates with the original layout before the charge, even on unwind.
    // No pointer escapes or unsafe Send/Sync implementation is needed.
    Ok(unsafe { Vec::from_raw_parts(pointer, 0, count) })
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Counts {
    n: usize,
    e: usize,
    d: usize,
    p: usize,
}
pub(super) struct SemanticPlan<'a> {
    bytes: &'a [u8],
    counts: Counts,
    layouts: [Layout; 4],
    requested: usize,
}
impl<'a> SemanticPlan<'a> {
    pub(super) fn scan(bytes: &'a [u8]) -> Result<Self, Error> {
        let mut sink = Sink { counts: Counts::default(), buffers: None, limits: Counts::default() };
        Cursor::new(bytes)?.document(&mut sink)?;
        let counts = sink.counts;
        let layouts = [
            layout::<Node>(counts.n, "nodes")?,
            layout::<Edge>(counts.e, "edges")?,
            layout::<u8>(counts.d, "text")?,
            layout::<u8>(counts.p, "numbers")?,
        ];
        let requested = add(add(layouts[0].size(), layouts[1].size())?, add(layouts[2].size(), layouts[3].size())?)?;
        Ok(Self { bytes, counts, layouts, requested })
    }
    pub(super) fn materialize(self, budget: &GgufDeclarationSourceBudget) -> Result<SemanticDocument, Error> {
        // Locals drop in reverse order: every initialized buffer precedes refund.
        let mut charge = budget.raw().charge();
        charge.acquire(self.requested).map_err(|source| Error::Admission { source })?;
        let nodes = allocate_exact::<Node>(self.counts.n, self.layouts[0], "nodes")?;
        let edges = allocate_exact::<Edge>(self.counts.e, self.layouts[1], "edges")?;
        let text = allocate_exact::<u8>(self.counts.d, self.layouts[2], "text")?;
        let numbers = allocate_exact::<u8>(self.counts.p, self.layouts[3], "numbers")?;
        let mut sink = Sink {
            counts: Counts::default(),
            limits: self.counts,
            buffers: Some(Buffers { nodes, edges, text, numbers }),
        };
        let root = Cursor::new(self.bytes)?.document(&mut sink)?;
        if sink.counts != self.counts {
            return Err(Error::PlanMismatch { storage: "counts", offset: self.bytes.len() });
        }
        let Some(Buffers { nodes, edges, text, numbers }) = sink.buffers else {
            return Err(Error::PlanMismatch { storage: "buffers", offset: 0 });
        };
        if [nodes.len(), edges.len(), text.len(), numbers.len()]
            != [self.counts.n, self.counts.e, self.counts.d, self.counts.p]
        {
            return Err(Error::PlanMismatch { storage: "lengths", offset: self.bytes.len() });
        }
        let source_sha256 = Sha256Digest::of(self.bytes);
        Ok(SemanticDocument { nodes, edges, text, numbers, root, source_sha256, _charge: charge })
    }
}

struct Buffers {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    text: Vec<u8>,
    numbers: Vec<u8>,
}
struct Sink {
    counts: Counts,
    limits: Counts,
    buffers: Option<Buffers>,
}
fn push<T>(vec: &mut Vec<T>, value: T, limit: usize, storage: &'static str, offset: usize) -> Result<(), Error> {
    if vec.len() >= limit || vec.len() >= vec.capacity() {
        return Err(Error::PlanMismatch { storage, offset });
    }
    vec.push(value);
    Ok(())
}
impl Sink {
    fn node(&mut self, offset: usize, value: NodeValue) -> Result<NodeId, Error> {
        let id = NodeId(self.counts.n);
        self.counts.n = add(self.counts.n, 1)?;
        if let Some(buffers) = &mut self.buffers {
            push(&mut buffers.nodes, Node { source_offset: offset, value }, self.limits.n, "nodes", offset)?;
        }
        Ok(id)
    }
    fn set(&mut self, id: NodeId, value: NodeValue) -> Result<(), Error> {
        if let Some(buffers) = &mut self.buffers {
            buffers.nodes.get_mut(id.0).ok_or(Error::PlanMismatch { storage: "node", offset: 0 })?.value = value;
        }
        Ok(())
    }
    fn text_char(&mut self, scalar: char, offset: usize) -> Result<(), Error> {
        let mut encoded = [0; 4];
        for byte in scalar.encode_utf8(&mut encoded).bytes() {
            self.counts.d = add(self.counts.d, 1)?;
            if let Some(buffers) = &mut self.buffers {
                push(&mut buffers.text, byte, self.limits.d, "text", offset)?;
            }
        }
        Ok(())
    }
    fn number_byte(&mut self, byte: u8, offset: usize) -> Result<(), Error> {
        self.counts.p = add(self.counts.p, 1)?;
        if let Some(buffers) = &mut self.buffers {
            push(&mut buffers.numbers, byte, self.limits.p, "numbers", offset)?;
        }
        Ok(())
    }
    fn attach(
        &mut self,
        frame: &mut Frame,
        key: Option<TextSpan>,
        key_offset: Option<usize>,
        value: NodeId,
    ) -> Result<(), Error> {
        let id = EdgeId(self.counts.e);
        self.counts.e = add(self.counts.e, 1)?;
        if let Some(buffers) = &mut self.buffers {
            push(
                &mut buffers.edges,
                Edge { next: None, key, key_offset, value },
                self.limits.e,
                "edges",
                key_offset.unwrap_or(0),
            )?;
            if let Some(key) = key {
                let mut next = frame.first;
                while let Some(previous) = next {
                    let edge =
                        buffers.edges.get_mut(previous.0).ok_or(Error::PlanMismatch { storage: "edge", offset: 0 })?;
                    if edge.key.is_some_and(|old| span_text(&buffers.text, old) == span_text(&buffers.text, key)) {
                        edge.value = value;
                        return Ok(());
                    }
                    next = edge.next;
                }
            }
            if let Some(tail) = frame.tail {
                buffers.edges.get_mut(tail.0).ok_or(Error::PlanMismatch { storage: "tail", offset: 0 })?.next =
                    Some(id);
            }
        }
        if frame.first.is_none() {
            frame.first = Some(id);
        }
        frame.tail = Some(id);
        frame.len = add(frame.len, 1)?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum State {
    First,
    Next,
    After,
}
#[derive(Clone, Copy)]
struct Frame {
    node: NodeId,
    object: bool,
    state: State,
    first: Option<EdgeId>,
    tail: Option<EdgeId>,
    len: usize,
    private: Option<NumberSpan>,
}
impl Frame {
    const EMPTY: Self =
        Self { node: NodeId(0), object: false, state: State::First, first: None, tail: None, len: 0, private: None };
}

struct Cursor<'a> {
    source: &'a str,
    offset: usize,
    frames: [Frame; 64],
    depth: usize,
}
impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self, Error> {
        let source = std::str::from_utf8(bytes)
            .map_err(|error| Error::Syntax { offset: error.valid_up_to(), reason: "invalid UTF-8" })?;
        Ok(Self { source, offset: 0, frames: [Frame::EMPTY; 64], depth: 0 })
    }
    fn error(&self, reason: &'static str) -> Error {
        Error::Syntax { offset: self.offset, reason }
    }
    fn peek(&self) -> Option<u8> {
        self.source.as_bytes().get(self.offset).copied()
    }
    fn advance(&mut self) -> Result<(), Error> {
        if self.peek().is_none() {
            return Err(self.error("unexpected end"));
        }
        self.offset = add(self.offset, 1)?;
        Ok(())
    }
    fn whitespace(&mut self) -> Result<(), Error> {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.advance()?;
        }
        Ok(())
    }
    fn required(&mut self, byte: u8) -> Result<(), Error> {
        if self.peek() != Some(byte) {
            return Err(self.error("unexpected token"));
        }
        self.advance()
    }
    fn stack(&mut self, frame: Option<Frame>) -> Result<(), Error> {
        if let Some(frame) = frame {
            if self.depth >= self.frames.len() {
                return Err(self.error("container depth exceeds envelope"));
            }
            self.frames[self.depth] = frame;
            self.depth = add(self.depth, 1)?;
        }
        Ok(())
    }
    fn document(mut self, sink: &mut Sink) -> Result<NodeId, Error> {
        self.whitespace()?;
        let (root, frame) = self.value(sink)?;
        self.stack(frame)?;
        while self.depth > 0 {
            self.whitespace()?;
            let top = self.depth - 1;
            let mut frame = self.frames[top];
            let close = if frame.object { b'}' } else { b']' };
            if matches!(frame.state, State::After) {
                if self.peek() != Some(close) {
                    if frame.private.is_some() {
                        return Err(self.error("private Number must have exactly one member"));
                    }
                    self.required(b',')?;
                    self.frames[top].state = State::Next;
                    continue;
                }
            } else if !(matches!(frame.state, State::First) && self.peek() == Some(close)) {
                let (key, key_offset, private) = if frame.object {
                    let offset = self.offset;
                    let (span, _, reserved) = self.string(sink)?;
                    self.whitespace()?;
                    self.required(b':')?;
                    self.whitespace()?;
                    (Some(span), Some(offset), matches!(frame.state, State::First) && reserved)
                } else {
                    (None, None, false)
                };
                let (id, child) = if private {
                    let offset = self.offset;
                    let (span, raw, _) = self.string(sink)?;
                    let id = sink.node(offset, NodeValue::String(span))?;
                    frame.private = Some(normalize_number(raw, sink, offset, true)?);
                    (id, None)
                } else {
                    self.value(sink)?
                };
                sink.attach(&mut frame, key, key_offset, id)?;
                frame.state = State::After;
                self.frames[top] = frame;
                self.stack(child)?;
                continue;
            }
            self.required(close)?;
            let value = match frame.private {
                Some(number) => NodeValue::Number(number),
                None if frame.object => NodeValue::Object { first: frame.first, len: frame.len },
                None => NodeValue::Array { first: frame.first, len: frame.len },
            };
            sink.set(frame.node, value)?;
            self.depth -= 1;
        }
        self.whitespace()?;
        if self.peek().is_some() {
            return Err(self.error("trailing input"));
        }
        Ok(root)
    }
    fn string(&mut self, sink: &mut Sink) -> Result<(TextSpan, Scalars<'a>, bool), Error> {
        self.required(b'"')?;
        let start = self.offset;
        let decoded_start = sink.counts.d;
        let mut scalars =
            Scalars { source: self.source, offset: start, end: self.source.len(), escaped: true, quoted: true };
        let mut reserved = "$serde_json::private::Number".chars();
        let mut equal = true;
        while let Some(scalar) = scalars.next()? {
            equal &= reserved.next() == Some(scalar);
            sink.text_char(scalar, scalars.offset)?;
        }
        let end = scalars.offset;
        self.offset = end;
        self.required(b'"')?;
        Ok((
            TextSpan { start: decoded_start, len: sink.counts.d - decoded_start },
            Scalars { source: self.source, offset: start, end, escaped: true, quoted: false },
            equal && reserved.next().is_none(),
        ))
    }
    fn value(&mut self, sink: &mut Sink) -> Result<(NodeId, Option<Frame>), Error> {
        let offset = self.offset;
        let byte = self.peek().ok_or_else(|| self.error("expected value"))?;
        let id = sink.node(offset, NodeValue::Null)?;
        let value = match byte {
            b'{' | b'[' => {
                self.advance()?;
                return Ok((id, Some(Frame { node: id, object: byte == b'{', ..Frame::EMPTY })));
            }
            b'"' => NodeValue::String(self.string(sink)?.0),
            b't' | b'f' | b'n' => {
                let literal: &[u8] = match byte {
                    b't' => b"true",
                    b'f' => b"false",
                    _ => b"null",
                };
                for expected in literal {
                    self.required(*expected)?;
                }
                match byte {
                    b't' => NodeValue::Bool(true),
                    b'f' => NodeValue::Bool(false),
                    _ => NodeValue::Null,
                }
            }
            b'-' | b'0'..=b'9' => {
                let raw =
                    Scalars { source: self.source, offset, end: self.source.len(), escaped: false, quoted: false };
                let (end, _) = scan_number(raw, offset, false)?;
                let raw = Scalars { end, ..raw };
                self.offset = end;
                NodeValue::Number(normalize_number(raw, sink, offset, false)?)
            }
            _ => return Err(self.error("expected value")),
        };
        sink.set(id, value)?;
        Ok((id, None))
    }
}

/// A borrowed decoder, also used for arbitrarily long private-Number strings.
#[derive(Clone, Copy)]
struct Scalars<'a> {
    source: &'a str,
    offset: usize,
    end: usize,
    escaped: bool,
    quoted: bool,
}
impl Scalars<'_> {
    fn error(&self, reason: &'static str) -> Error {
        Error::Syntax { offset: self.offset, reason }
    }
    fn peek(self) -> Result<Option<char>, Error> {
        let mut copy = self;
        copy.next()
    }
    fn byte(&mut self) -> Result<u8, Error> {
        let byte = self
            .source
            .as_bytes()
            .get(self.offset)
            .copied()
            .filter(|_| self.offset < self.end)
            .ok_or_else(|| self.error("truncated escape"))?;
        self.offset = add(self.offset, 1)?;
        Ok(byte)
    }
    fn hex(&mut self) -> Result<u32, Error> {
        let mut value = 0;
        for _ in 0..4 {
            let digit = self.byte()?;
            value = (value << 4)
                | match digit {
                    b'0'..=b'9' => u32::from(digit - b'0'),
                    b'a'..=b'f' => u32::from(digit - b'a' + 10),
                    b'A'..=b'F' => u32::from(digit - b'A' + 10),
                    _ => return Err(self.error("invalid hex escape")),
                };
        }
        Ok(value)
    }
    fn next(&mut self) -> Result<Option<char>, Error> {
        if self.offset == self.end {
            return if self.quoted { Err(self.error("unterminated string")) } else { Ok(None) };
        }
        let scalar = self
            .source
            .get(self.offset..self.end)
            .and_then(|tail| tail.chars().next())
            .ok_or_else(|| self.error("invalid scalar boundary"))?;
        if self.quoted && scalar == '"' {
            return Ok(None);
        }
        if self.escaped && scalar <= '\u{1f}' {
            return Err(self.error("unescaped control"));
        }
        self.offset = add(self.offset, scalar.len_utf8())?;
        if !self.escaped || scalar != '\\' {
            return Ok(Some(scalar));
        }
        let escape = self.byte()?;
        Ok(Some(match escape {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{8}',
            b'f' => '\u{c}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => {
                let high = self.hex()?;
                let code = if (0xd800..=0xdbff).contains(&high) {
                    if self.byte()? != b'\\' || self.byte()? != b'u' {
                        return Err(self.error("missing low surrogate"));
                    }
                    let low = self.hex()?;
                    if !(0xdc00..=0xdfff).contains(&low) {
                        return Err(self.error("invalid low surrogate"));
                    }
                    0x10000 + ((high - 0xd800) << 10) + (low - 0xdc00)
                } else {
                    high
                };
                char::from_u32(code).ok_or_else(|| self.error("invalid Unicode escape"))?
            }
            _ => return Err(self.error("invalid escape")),
        }))
    }
}

fn scan_number(mut input: Scalars<'_>, offset: usize, entire: bool) -> Result<(usize, bool), Error> {
    let error = || Error::Number { offset, reason: "expected a complete JSON number" };
    let negative = input.peek()? == Some('-');
    if negative {
        input.next()?;
    }
    let first = input.next()?.ok_or_else(error)?;
    if !first.is_ascii_digit() {
        return Err(error());
    }
    let mut magnitude = Some(u64::from(first as u8 - b'0'));
    while input.peek()?.is_some_and(|ch| ch.is_ascii_digit()) {
        if first == '0' {
            return Err(error());
        }
        let digit = input.next()?.ok_or_else(error)? as u8 - b'0';
        // Magnitude overflow is valid arbitrary precision, never storage overflow.
        magnitude = magnitude.and_then(|old| old.checked_mul(10)?.checked_add(u64::from(digit)));
    }
    let mut integer = true;
    if input.peek()? == Some('.') {
        integer = false;
        input.next()?;
        if !input.peek()?.is_some_and(|ch| ch.is_ascii_digit()) {
            return Err(error());
        }
        while input.peek()?.is_some_and(|ch| ch.is_ascii_digit()) {
            input.next()?;
        }
    }
    if matches!(input.peek()?, Some('e' | 'E')) {
        integer = false;
        input.next()?;
        if matches!(input.peek()?, Some('+' | '-')) {
            input.next()?;
        }
        if !input.peek()?.is_some_and(|ch| ch.is_ascii_digit()) {
            return Err(error());
        }
        while input.peek()?.is_some_and(|ch| ch.is_ascii_digit()) {
            input.next()?;
        }
    }
    if entire && input.peek()?.is_some() {
        return Err(error());
    }
    let fits = magnitude.is_some_and(|m| !negative || m <= (i64::MAX as u64) + 1);
    Ok((input.offset, integer && fits && negative && magnitude == Some(0)))
}
fn normalize_number(mut input: Scalars<'_>, sink: &mut Sink, offset: usize, entire: bool) -> Result<NumberSpan, Error> {
    let (_, negative_zero) = scan_number(input, offset, entire)?;
    let start = sink.counts.p;
    if negative_zero {
        sink.number_byte(b'0', offset)?;
    } else {
        // Valid JSON integer spellings already are canonical, except -0 above.
        while let Some(ch) = input.next()? {
            if !ch.is_ascii() {
                return Err(Error::Number { offset, reason: "non-ASCII number" });
            }
            let byte = ch as u8;
            sink.number_byte(if byte == b'E' { b'e' } else { byte }, offset)?;
            if matches!(byte, b'e' | b'E') && !matches!(input.peek()?, Some('+' | '-')) {
                sink.number_byte(b'+', offset)?;
            }
        }
    }
    Ok(NumberSpan { start, len: sink.counts.p - start })
}
