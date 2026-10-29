//! Support for the [GGUF file format](https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md).
//!
//! Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

use super::{GgmlDType, QTensor};
use crate::{Context, Device, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;
use std::io::{Read, Seek};

pub const DEFAULT_ALIGNMENT: u64 = 32;

// Caps mirror ggml-org/llama.cpp#19856 (GGUF_MAX_STRING_LENGTH,
// GGUF_MAX_ARRAY_ELEMENTS) and GGML_MAX_DIMS from ggml.h. The depth cap
// bounds recursion through nested `Value::Array` so a crafted file cannot
// overflow the stack with a chain of arrays-of-arrays.
const GGUF_MAX_STRING_LENGTH: u64 = 1 << 30;
const GGUF_MAX_ARRAY_ELEMENTS: u64 = 1 << 30;
const GGUF_MAX_TENSOR_DIMS: u32 = 4;
const GGUF_MAX_VALUE_DEPTH: usize = 64;

/// Explicit quotas for [`Content::read_with_limits`]. Every field is finite;
/// zero grants no allowance for that dimension. The legacy reader ignores them.
/// Existing format caps still apply when a caller supplies larger limits.
///
/// The allocation ledger counts planned requested storage cumulatively, without
/// refunds: raw strings, up to three output bytes per raw value-string byte,
/// array slots, dimension vectors and map tables. Each nonempty buffer/table
/// adds 64 bytes. Maps charge `4 * entries + 16` buckets, each containing the
/// key/value tuple plus one control byte (assuming the current std hash-table
/// layout, with conservative capacity/control/alignment slack).
/// Stack values, error diagnostics and allocations inside the supplied reader
/// are outside the ledger. Allocator metadata, rounding, overcommit and unrelated
/// process memory are not bounded: this is not a physical-memory/OOM guarantee.
#[derive(Debug, Clone, Copy)]
pub struct ReadLimits {
    /// Raw bytes consumed from the initial reader position, excluding padding/payload.
    pub max_header_bytes: u64,
    /// Number of metadata entries.
    pub max_metadata_entries: u64,
    /// Number of tensor inventory entries.
    pub max_tensors: u64,
    /// Raw byte length of any key, name or value string.
    pub max_string_bytes: u64,
    /// Sum of raw byte lengths of all keys, names and value strings.
    pub max_total_string_bytes: u64,
    /// All metadata Value nodes, including array containers and every element.
    pub max_total_values: u64,
    /// Number of enclosing array levels; zero permits only scalar/string values.
    pub max_nesting: usize,
    /// Number of dimensions per tensor; zero permits only rank-zero inventory entries.
    pub max_rank: u32,
    /// Cumulative allocation-ledger bytes, using the accounting above.
    pub max_allocation_bytes: u64,
}

impl Default for ReadLimits {
    /// Conservative G1 header defaults, not measured machine capacity or model admission.
    fn default() -> Self {
        Self {
            max_header_bytes: 64 << 20,
            max_metadata_entries: 4096,
            max_tensors: 8195,
            max_string_bytes: 1 << 20,
            max_total_string_bytes: 32 << 20,
            max_total_values: 1 << 20,
            max_nesting: 2,
            max_rank: 4,
            max_allocation_bytes: 256 << 20,
        }
    }
}

struct ReadBudget<'a> {
    limits: &'a ReadLimits,
    strings: u64,
    values: u64,
    allocations: u64,
}

impl ReadBudget<'_> {
    fn charge(total: &mut u64, amount: u64, limit: u64, what: &str) -> Result<()> {
        let next = total
            .checked_add(amount)
            .with_context(|| format!("gguf: {what} accounting overflows u64"))?;
        if next > limit {
            crate::bail!("gguf: {what} {next} exceeds limit {limit}")
        }
        *total = next;
        Ok(())
    }

    fn charge_values(&mut self, count: u64) -> Result<()> {
        Self::charge(
            &mut self.values,
            count,
            self.limits.max_total_values,
            "cumulative value nodes",
        )
    }

    fn charge_storage(&mut self, count: u64, item_size: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let item_size = u64::try_from(item_size).context("gguf: item size does not fit u64")?;
        let bytes = count
            .checked_mul(item_size)
            .and_then(|n| n.checked_add(64))
            .context("gguf: allocation accounting overflows u64")?;
        Self::charge(
            &mut self.allocations,
            bytes,
            self.limits.max_allocation_bytes,
            "cumulative allocation ledger",
        )
    }

    fn reserve_map<K, V>(&mut self, map: &mut HashMap<K, V>, count: u64) -> Result<()>
    where
        K: Eq + std::hash::Hash,
    {
        if count == 0 {
            return Ok(());
        }
        let buckets = count
            .checked_mul(4)
            .and_then(|n| n.checked_add(16))
            .context("gguf: map bucket accounting overflows u64")?;
        let bucket_size = std::mem::size_of::<(K, V)>()
            .checked_add(1)
            .context("gguf: map bucket size overflows usize")?;
        self.charge_storage(buckets, bucket_size)?;
        let count = usize::try_from(count).context("gguf: map count does not fit usize")?;
        map.try_reserve(count)
            .context("gguf: cannot allocate header map")?;
        Ok(())
    }
}

// Only header reads pass through this wrapper. Sizing/position seeks do not
// consume the relative byte budget, and no payload is read here.
struct HeaderReader<'a, R> {
    inner: &'a mut R,
    budget: Option<ReadBudget<'a>>,
    consumed: u64,
    position: u64,
    file_size: u64,
}

impl<R> HeaderReader<'_, R> {
    fn check_available(&self, bytes: u64) -> std::io::Result<()> {
        if let Some(budget) = &self.budget {
            let total = self.consumed.checked_add(bytes).ok_or_else(|| {
                std::io::Error::other("gguf: header byte accounting overflows u64")
            })?;
            if total > budget.limits.max_header_bytes {
                return Err(std::io::Error::other(
                    "gguf: raw header byte limit exceeded",
                ));
            }
            let remaining = self
                .file_size
                .checked_sub(self.position)
                .ok_or_else(|| std::io::Error::other("gguf: reader position exceeds file size"))?;
            if bytes > remaining {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "gguf: truncated header exceeds remaining file bytes",
                ));
            }
        }
        Ok(())
    }

    fn record_read(&mut self, bytes: u64) -> std::io::Result<()> {
        if self.budget.is_some() {
            // check_available has already admitted the complete request before
            // the read; retain checked arithmetic for both accounting units.
            self.consumed = self.consumed.checked_add(bytes).ok_or_else(|| {
                std::io::Error::other("gguf: header byte accounting overflows u64")
            })?;
            self.position = self
                .position
                .checked_add(bytes)
                .ok_or_else(|| std::io::Error::other("gguf: header position overflows u64"))?;
        }
        Ok(())
    }
}

impl<R: std::io::Read> std::io::Read for HeaderReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let bytes = u64::try_from(buf.len())
            .map_err(|_| std::io::Error::other("gguf: read length does not fit u64"))?;
        self.check_available(bytes)?;
        let n = self.inner.read(buf)?;
        let bytes = u64::try_from(n)
            .map_err(|_| std::io::Error::other("gguf: read length does not fit u64"))?;
        self.record_read(bytes)?;
        Ok(n)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        let bytes = u64::try_from(buf.len())
            .map_err(|_| std::io::Error::other("gguf: read length does not fit u64"))?;
        self.check_available(bytes)?;
        self.inner.read_exact(buf)?;
        self.record_read(bytes)?;
        Ok(())
    }
}

impl<R: std::io::Seek> std::io::Seek for HeaderReader<'_, R> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let position = self.inner.seek(pos)?;
        self.position = position;
        Ok(position)
    }

    fn stream_position(&mut self) -> std::io::Result<u64> {
        let position = self.inner.stream_position()?;
        self.position = position;
        Ok(position)
    }
}

// `file_size` is the byte length captured once up front, so this avoids
// seeking to the end and back on every length-prefixed read.
fn remaining_bytes<R: std::io::Seek>(reader: &mut R, file_size: u64) -> Result<u64> {
    let cur = reader.stream_position()?;
    file_size
        .checked_sub(cur)
        .context("gguf: reader position exceeds file size")
}

fn read_length<R: std::io::Read>(reader: &mut R, magic: &VersionedMagic) -> Result<u64> {
    match magic {
        VersionedMagic::GgufV1 => Ok(reader.read_u32::<LittleEndian>()? as u64),
        VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => Ok(reader.read_u64::<LittleEndian>()?),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Magic {
    Gguf,
}

impl TryFrom<u32> for Magic {
    type Error = crate::Error;
    fn try_from(value: u32) -> Result<Self> {
        let magic = match value {
            0x46554747 | 0x47475546 => Self::Gguf,
            _ => crate::bail!("unknown magic 0x{value:08x}"),
        };
        Ok(magic)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgufV1,
    GgufV2,
    GgufV3,
}

impl VersionedMagic {
    fn read<R: std::io::Read>(reader: &mut R) -> Result<Self> {
        let magic = reader.read_u32::<LittleEndian>()?;
        let magic = Magic::try_from(magic)?;
        let version = reader.read_u32::<LittleEndian>()?;
        let versioned_magic = match (magic, version) {
            (Magic::Gguf, 1) => Self::GgufV1,
            (Magic::Gguf, 2) => Self::GgufV2,
            (Magic::Gguf, 3) => Self::GgufV3,
            _ => crate::bail!("gguf: unsupported magic/version {magic:?}/{version}"),
        };
        Ok(versioned_magic)
    }

    fn length_prefix_size(&self) -> u64 {
        match self {
            Self::GgufV1 => 4,
            Self::GgufV2 | Self::GgufV3 => 8,
        }
    }
}

#[derive(Debug)]
pub struct TensorInfo {
    pub ggml_dtype: GgmlDType,
    pub shape: crate::Shape,
    pub offset: u64,
}

impl TensorInfo {
    // Validate inventory and public reads before allocating. Keep the same dimension
    // order, including intermediate products for shapes containing zero.
    fn size_in_bytes(&self) -> Result<usize> {
        let tensor_elems = self.shape.dims().iter().try_fold(1usize, |n, &dim| {
            n.checked_mul(dim)
                .context("gguf: tensor element count overflows usize")
        })?;
        let block_size = self.ggml_dtype.block_size();
        if !tensor_elems.is_multiple_of(block_size) {
            crate::bail!("gguf: tensor element count {tensor_elems} is not divisible by block size {block_size}")
        }
        if let Some(&last_dim) = self.shape.dims().last() {
            if !last_dim.is_multiple_of(block_size) {
                crate::bail!("gguf: tensor last dimension {last_dim} is not divisible by block size {block_size}")
            }
        }
        (tensor_elems / block_size)
            .checked_mul(self.ggml_dtype.type_size())
            .context("gguf: tensor byte length overflows usize")
    }

    fn byte_range(&self, tensor_data_offset: u64, file_size: u64) -> Result<(u64, usize)> {
        let size_in_bytes = self.size_in_bytes()?;
        let byte_len =
            u64::try_from(size_in_bytes).context("gguf: tensor byte length does not fit u64")?;
        let tensor_start = tensor_data_offset
            .checked_add(self.offset)
            .context("gguf: tensor start offset overflows u64")?;
        let tensor_end = tensor_start
            .checked_add(byte_len)
            .context("gguf: tensor end offset overflows u64")?;
        if tensor_end > file_size {
            crate::bail!("gguf: tensor needs {size_in_bytes} bytes at offset {tensor_start}, file size is {file_size}")
        }
        Ok((tensor_start, size_in_bytes))
    }

    pub fn read<R: std::io::Seek + std::io::Read>(
        &self,
        reader: &mut R,
        tensor_data_offset: u64,
        device: &Device,
    ) -> Result<QTensor> {
        let file_size = reader.seek(std::io::SeekFrom::End(0))?;
        let (tensor_start, size_in_bytes) = self.byte_range(tensor_data_offset, file_size)?;
        // QTensor does not support scalars, even for unquantized GGML dtypes.
        if self.shape.dims().is_empty() {
            crate::bail!("gguf: scalar tensor cannot be quantized")
        }
        if size_in_bytes == 0 {
            crate::bail!("gguf: empty tensor cannot be materialized")
        }
        let mut raw_data = Vec::new();
        raw_data
            .try_reserve_exact(size_in_bytes)
            .context("gguf: cannot allocate tensor read buffer")?;
        raw_data.resize(size_in_bytes, 0u8);
        reader.seek(std::io::SeekFrom::Start(tensor_start))?;
        reader.read_exact(&mut raw_data)?;
        super::ggml_file::qtensor_from_ggml(
            self.ggml_dtype,
            &raw_data,
            self.shape.dims().to_vec(),
            device,
        )
    }
}

#[derive(Debug)]
pub struct Content {
    pub magic: VersionedMagic,
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, TensorInfo>,
    pub tensor_data_offset: u64,
}

fn read_string<R: std::io::Read + std::io::Seek>(
    reader: &mut HeaderReader<'_, R>,
    magic: &VersionedMagic,
    file_size: u64,
    strict_name: bool,
) -> Result<String> {
    let len = read_length(reader, magic)?;
    if len > GGUF_MAX_STRING_LENGTH {
        crate::bail!("gguf: string length {len} exceeds max {GGUF_MAX_STRING_LENGTH}")
    }
    let remaining = remaining_bytes(reader, file_size)?;
    if len > remaining {
        crate::bail!("gguf: string length {len} exceeds remaining file bytes {remaining}")
    }
    if reader.budget.is_some() {
        reader.check_available(len)?;
        return read_bounded_string(reader, len, strict_name);
    }
    let len = usize::try_from(len).context("gguf: string length does not fit usize")?;
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .context("gguf: cannot allocate string read buffer")?;
    v.resize(len, 0u8);
    reader.read_exact(&mut v)?;
    // GGUF strings are supposed to be non-null terminated but in practice this happens.
    while let Some(0) = v.last() {
        v.pop();
    }
    // GGUF strings are utf8 encoded but there are cases that don't seem to be valid.
    Ok(String::from_utf8_lossy(&v).into_owned())
}

fn read_bounded_string<R: std::io::Read>(
    reader: &mut HeaderReader<'_, R>,
    len: u64,
    strict_name: bool,
) -> Result<String> {
    // Called only with a budget; raw and output storage are charged before either
    // reservation/read. No infallibly allocating lossy-decoding helper is used.
    let output_bytes = if strict_name {
        0
    } else {
        len.checked_mul(3)
            .context("gguf: lossy string size overflows u64")?
    };
    if let Some(budget) = &mut reader.budget {
        if len > budget.limits.max_string_bytes {
            crate::bail!(
                "gguf: string length {len} exceeds limit {}",
                budget.limits.max_string_bytes
            )
        }
        ReadBudget::charge(
            &mut budget.strings,
            len,
            budget.limits.max_total_string_bytes,
            "cumulative raw string bytes",
        )?;
        budget.charge_storage(len, 1)?;
        budget.charge_storage(output_bytes, 1)?;
    }
    let len = usize::try_from(len).context("gguf: string length does not fit usize")?;
    let output_bytes =
        usize::try_from(output_bytes).context("gguf: decoded string size does not fit usize")?;
    let mut raw = Vec::new();
    raw.try_reserve_exact(len)
        .context("gguf: cannot allocate string read buffer")?;
    raw.resize(len, 0u8);
    reader.read_exact(&mut raw)?;
    if strict_name {
        if raw.contains(&0) {
            crate::bail!("gguf: metadata key or tensor name contains NUL")
        }
        return String::from_utf8(raw).context("gguf: metadata key or tensor name is not UTF-8");
    }
    while let Some(0) = raw.last() {
        raw.pop();
    }
    let mut decoded = String::new();
    decoded
        .try_reserve_exact(output_bytes)
        .context("gguf: cannot allocate decoded value string")?;
    for chunk in raw.utf8_chunks() {
        decoded.push_str(chunk.valid());
        if !chunk.invalid().is_empty() {
            decoded.push('\u{fffd}');
        }
    }
    Ok(decoded)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    // The value is a 8-bit unsigned integer.
    U8,
    // The value is a 8-bit signed integer.
    I8,
    // The value is a 16-bit unsigned little-endian integer.
    U16,
    // The value is a 16-bit signed little-endian integer.
    I16,
    // The value is a 32-bit unsigned little-endian integer.
    U32,
    // The value is a 32-bit signed little-endian integer.
    I32,
    // The value is a 64-bit unsigned little-endian integer.
    U64,
    // The value is a 64-bit signed little-endian integer.
    I64,
    // The value is a 32-bit IEEE754 floating point number.
    F32,
    // The value is a 64-bit IEEE754 floating point number.
    F64,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    Bool,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    String,
    // The value is an array of other values, with the length and type prepended.
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    Array,
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn value_type(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F32(_) => ValueType::F32,
            Self::F64(_) => ValueType::F64,
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::Array(_) => ValueType::Array,
        }
    }

    pub fn to_u8(&self) -> Result<u8> {
        match self {
            Self::U8(v) => Ok(*v),
            v => crate::bail!("not a u8 {v:?}"),
        }
    }

    pub fn to_i8(&self) -> Result<i8> {
        match self {
            Self::I8(v) => Ok(*v),
            v => crate::bail!("not a i8 {v:?}"),
        }
    }

    pub fn to_u16(&self) -> Result<u16> {
        match self {
            Self::U16(v) => Ok(*v),
            v => crate::bail!("not a u16 {v:?}"),
        }
    }

    pub fn to_i16(&self) -> Result<i16> {
        match self {
            Self::I16(v) => Ok(*v),
            v => crate::bail!("not a i16 {v:?}"),
        }
    }

    pub fn to_u32(&self) -> Result<u32> {
        match self {
            Self::U32(v) => Ok(*v),
            v => crate::bail!("not a u32 {v:?}"),
        }
    }

    pub fn to_i32(&self) -> Result<i32> {
        match self {
            Self::I32(v) => Ok(*v),
            v => crate::bail!("not a i32 {v:?}"),
        }
    }

    /// This will also automatically upcast any integral types which will not truncate.
    pub fn to_u64(&self) -> Result<u64> {
        match self {
            Self::U64(v) => Ok(*v),
            // Autoupcast cases here
            Self::U8(v) => Ok(*v as u64),
            Self::U16(v) => Ok(*v as u64),
            Self::U32(v) => Ok(*v as u64),
            Self::Bool(v) => Ok(*v as u64),
            v => crate::bail!("not a u64 or upcastable to u64 {v:?}"),
        }
    }

    pub fn to_i64(&self) -> Result<i64> {
        match self {
            Self::I64(v) => Ok(*v),
            v => crate::bail!("not a i64 {v:?}"),
        }
    }

    pub fn to_f32(&self) -> Result<f32> {
        match self {
            Self::F32(v) => Ok(*v),
            v => crate::bail!("not a f32 {v:?}"),
        }
    }

    pub fn to_f64(&self) -> Result<f64> {
        match self {
            Self::F64(v) => Ok(*v),
            v => crate::bail!("not a f64 {v:?}"),
        }
    }

    pub fn to_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            v => crate::bail!("not a bool {v:?}"),
        }
    }

    pub fn to_vec(&self) -> Result<&Vec<Value>> {
        match self {
            Self::Array(v) => Ok(v),
            v => crate::bail!("not a vec {v:?}"),
        }
    }

    pub fn to_string(&self) -> Result<&String> {
        match self {
            Self::String(v) => Ok(v),
            v => crate::bail!("not a string {v:?}"),
        }
    }

    fn read<R: std::io::Read + std::io::Seek>(
        reader: &mut HeaderReader<'_, R>,
        value_type: ValueType,
        magic: &VersionedMagic,
        depth: usize,
        file_size: u64,
    ) -> Result<Self> {
        if depth > GGUF_MAX_VALUE_DEPTH {
            crate::bail!("gguf: value nesting depth exceeds max {GGUF_MAX_VALUE_DEPTH}")
        }
        let v = match value_type {
            ValueType::U8 => Self::U8(reader.read_u8()?),
            ValueType::I8 => Self::I8(reader.read_i8()?),
            ValueType::U16 => Self::U16(reader.read_u16::<LittleEndian>()?),
            ValueType::I16 => Self::I16(reader.read_i16::<LittleEndian>()?),
            ValueType::U32 => Self::U32(reader.read_u32::<LittleEndian>()?),
            ValueType::I32 => Self::I32(reader.read_i32::<LittleEndian>()?),
            ValueType::U64 => Self::U64(reader.read_u64::<LittleEndian>()?),
            ValueType::I64 => Self::I64(reader.read_i64::<LittleEndian>()?),
            ValueType::F32 => Self::F32(reader.read_f32::<LittleEndian>()?),
            ValueType::F64 => Self::F64(reader.read_f64::<LittleEndian>()?),
            ValueType::Bool => match reader.read_u8()? {
                0 => Self::Bool(false),
                1 => Self::Bool(true),
                b => crate::bail!("unexpected bool value {b}"),
            },
            ValueType::String => Self::String(read_string(reader, magic, file_size, false)?),
            ValueType::Array => {
                if let Some(budget) = &reader.budget {
                    if depth >= budget.limits.max_nesting {
                        crate::bail!(
                            "gguf: array nesting exceeds limit {}",
                            budget.limits.max_nesting
                        )
                    }
                }
                let value_type = reader.read_u32::<LittleEndian>()?;
                let value_type = ValueType::from_u32(value_type)?;
                let len = read_length(reader, magic)?;
                if len > GGUF_MAX_ARRAY_ELEMENTS {
                    crate::bail!("gguf: array length {len} exceeds max {GGUF_MAX_ARRAY_ELEMENTS}")
                }
                // The capped length and the at-most-12-byte minimum fit u64.
                let needed = len * value_type.min_disk_size(magic);
                let remaining = remaining_bytes(reader, file_size)?;
                if needed > remaining {
                    crate::bail!(
                        "gguf: array of {len} elements needs at least {needed} bytes, only {remaining} remaining"
                    )
                }
                let mut vs = Vec::new();
                if let Some(budget) = &mut reader.budget {
                    // Parent/root slots have already been charged. Charge every
                    // child once, including children that are themselves arrays.
                    budget.charge_values(len)?;
                    budget.charge_storage(len, std::mem::size_of::<Value>())?;
                    reader.check_available(needed)?;
                    let len =
                        usize::try_from(len).context("gguf: array length does not fit usize")?;
                    vs.try_reserve_exact(len)
                        .context("gguf: cannot allocate value array")?;
                }
                for _ in 0..len {
                    vs.push(Value::read(
                        reader,
                        value_type,
                        magic,
                        depth + 1,
                        file_size,
                    )?)
                }
                Self::Array(vs)
            }
        };
        Ok(v)
    }

    fn write<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        match self {
            &Self::U8(v) => w.write_u8(v)?,
            &Self::I8(v) => w.write_i8(v)?,
            &Self::U16(v) => w.write_u16::<LittleEndian>(v)?,
            &Self::I16(v) => w.write_i16::<LittleEndian>(v)?,
            &Self::U32(v) => w.write_u32::<LittleEndian>(v)?,
            &Self::I32(v) => w.write_i32::<LittleEndian>(v)?,
            &Self::U64(v) => w.write_u64::<LittleEndian>(v)?,
            &Self::I64(v) => w.write_i64::<LittleEndian>(v)?,
            &Self::F32(v) => w.write_f32::<LittleEndian>(v)?,
            &Self::F64(v) => w.write_f64::<LittleEndian>(v)?,
            &Self::Bool(v) => w.write_u8(u8::from(v))?,
            Self::String(v) => write_string(w, v.as_str())?,
            Self::Array(v) => {
                // The `Value` type does not enforce that all the values in an Array have the same
                // type.
                let value_type = if v.is_empty() {
                    // Doesn't matter, the array is empty.
                    ValueType::U32
                } else {
                    let value_type: std::collections::HashSet<_> =
                        v.iter().map(|elem| elem.value_type()).collect();
                    if value_type.len() != 1 {
                        crate::bail!("multiple value-types in the same array {value_type:?}")
                    }
                    value_type.into_iter().next().context("empty value_type")?
                };
                w.write_u32::<LittleEndian>(value_type.to_u32())?;
                w.write_u64::<LittleEndian>(v.len() as u64)?;
                for elem in v.iter() {
                    elem.write(w)?
                }
            }
        }
        Ok(())
    }
}

impl ValueType {
    fn from_u32(v: u32) -> Result<Self> {
        let v = match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            v => crate::bail!("unrecognized value-type {v:#08x}"),
        };
        Ok(v)
    }

    fn to_u32(self) -> u32 {
        match self {
            Self::U8 => 0,
            Self::I8 => 1,
            Self::U16 => 2,
            Self::I16 => 3,
            Self::U32 => 4,
            Self::I32 => 5,
            Self::F32 => 6,
            Self::Bool => 7,
            Self::String => 8,
            Self::Array => 9,
            Self::U64 => 10,
            Self::I64 => 11,
            Self::F64 => 12,
        }
    }

    /// Minimum on-disk size of one value of this type, used to reject array
    /// lengths that exceed the remaining file size before allocating.
    fn min_disk_size(&self, magic: &VersionedMagic) -> u64 {
        match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
            Self::String => magic.length_prefix_size(),
            Self::Array => 4 + magic.length_prefix_size(),
        }
    }
}

impl Content {
    pub fn read<R: std::io::Seek + std::io::Read>(reader: &mut R) -> Result<Self> {
        Self::read_impl(reader, None)
    }

    /// Read only the GGUF header/inventory under explicit cumulative quotas.
    ///
    /// Checks precede input reads and planned collection/buffer reservations.
    /// Keys and tensor names must be UTF-8 without NUL; duplicates are rejected.
    /// Value strings retain the legacy trailing-NUL removal and lossy UTF-8
    /// decoding. The byte allowance starts at the current reader position;
    /// file-sizing seeks, final alignment padding and payload are not consumed.
    ///
    /// This retains generic v1/v2/v3 format compatibility and existing span and
    /// alignment checks. It does not enforce a model's inventory, canonical
    /// alignment, immutable source, payload quality or materialization budget.
    /// See [`ReadLimits`] for ledger assumptions and physical-memory exclusions.
    pub fn read_with_limits<R: std::io::Seek + std::io::Read>(
        reader: &mut R,
        limits: &ReadLimits,
    ) -> Result<Self> {
        Self::read_impl(reader, Some(limits))
    }

    fn read_impl<R: std::io::Seek + std::io::Read>(
        reader: &mut R,
        limits: Option<&ReadLimits>,
    ) -> Result<Self> {
        // Capture the file size once so the bounds checks below don't have to
        // seek to the end and back on every length-prefixed read.
        let start = reader.stream_position()?;
        let file_size = reader.seek(std::io::SeekFrom::End(0))?;
        reader.seek(std::io::SeekFrom::Start(start))?;

        let reader = &mut HeaderReader {
            inner: reader,
            budget: limits.map(|limits| ReadBudget {
                limits,
                strings: 0,
                values: 0,
                allocations: 0,
            }),
            consumed: 0,
            position: start,
            file_size,
        };
        let magic = VersionedMagic::read(reader)?;
        let tensor_count = read_length(reader, &magic)?;
        let metadata_kv_count = read_length(reader, &magic)?;

        if tensor_count > GGUF_MAX_ARRAY_ELEMENTS {
            crate::bail!("gguf: tensor_count {tensor_count} exceeds max {GGUF_MAX_ARRAY_ELEMENTS}")
        }
        if metadata_kv_count > GGUF_MAX_ARRAY_ELEMENTS {
            crate::bail!(
                "gguf: metadata_kv_count {metadata_kv_count} exceeds max {GGUF_MAX_ARRAY_ELEMENTS}"
            )
        }
        if let Some(budget) = &reader.budget {
            if tensor_count > budget.limits.max_tensors {
                crate::bail!(
                    "gguf: tensor count {tensor_count} exceeds limit {}",
                    budget.limits.max_tensors
                )
            }
            if metadata_kv_count > budget.limits.max_metadata_entries {
                crate::bail!(
                    "gguf: metadata count {metadata_kv_count} exceeds limit {}",
                    budget.limits.max_metadata_entries
                )
            }
        }

        // Reject header-declared counts that can't fit in the file at minimum size.
        // Per-entry minima: a metadata kv is at least `key_len_prefix + u32 value_type
        // + 1 byte value`; a tensor info is at least `name_len_prefix + u32 n_dims
        // + u32 dtype + u64 offset`.
        let prefix = magic.length_prefix_size();
        let min_per_kv = prefix + 4 + 1;
        let min_per_tensor = prefix + 4 + 4 + 8;
        // Both counts are capped at 2^30 and each minimum is at most 24 bytes.
        let needed = metadata_kv_count * min_per_kv + tensor_count * min_per_tensor;
        let remaining = remaining_bytes(reader, file_size)?;
        if needed > remaining {
            crate::bail!(
                "gguf: header declares {tensor_count} tensors and {metadata_kv_count} metadata entries, needs at least {needed} bytes, only {remaining} remaining"
            )
        }

        let mut metadata = HashMap::new();
        reader.check_available(needed)?;
        if let Some(budget) = &mut reader.budget {
            budget.charge_values(metadata_kv_count)?;
            budget.reserve_map(&mut metadata, metadata_kv_count)?;
        }
        for _idx in 0..metadata_kv_count {
            let key = read_string(reader, &magic, file_size, true)?;
            // Legacy names retain trailing-NUL removal and lossy UTF-8 decoding;
            // explicitly bounded names reach this check without normalization.
            if metadata.contains_key(&key) {
                crate::bail!("gguf: duplicate metadata key '{key}'")
            }
            let value_type = reader.read_u32::<LittleEndian>()?;
            let value_type = ValueType::from_u32(value_type)?;
            let value = Value::read(reader, value_type, &magic, 0, file_size)?;
            metadata.insert(key, value);
        }
        let mut tensor_infos = HashMap::new();
        let tensor_header_bytes = tensor_count
            .checked_mul(min_per_tensor)
            .context("gguf: tensor header size overflows u64")?;
        reader.check_available(tensor_header_bytes)?;
        if let Some(budget) = &mut reader.budget {
            budget.reserve_map(&mut tensor_infos, tensor_count)?;
        }
        for _idx in 0..tensor_count {
            let tensor_name = read_string(reader, &magic, file_size, true)?;
            if tensor_infos.contains_key(&tensor_name) {
                crate::bail!("gguf: duplicate tensor name '{tensor_name}'")
            }
            let n_dimensions = reader.read_u32::<LittleEndian>()?;
            if n_dimensions > GGUF_MAX_TENSOR_DIMS {
                crate::bail!(
                    "gguf: tensor '{tensor_name}' has {n_dimensions} dimensions, max is {GGUF_MAX_TENSOR_DIMS}"
                )
            }

            let mut dimensions = if let Some(budget) = &mut reader.budget {
                if n_dimensions > budget.limits.max_rank {
                    crate::bail!(
                        "gguf: tensor '{tensor_name}' rank {n_dimensions} exceeds limit {}",
                        budget.limits.max_rank
                    )
                }
                budget.charge_storage(u64::from(n_dimensions), std::mem::size_of::<usize>())?;
                let dimension_bytes = u64::from(n_dimensions)
                    .checked_mul(prefix)
                    .context("gguf: dimension byte count overflows u64")?;
                reader.check_available(dimension_bytes)?;
                let n_dimensions = usize::try_from(n_dimensions)
                    .context("gguf: tensor rank does not fit usize")?;
                let mut dimensions = Vec::new();
                dimensions
                    .try_reserve_exact(n_dimensions)
                    .context("gguf: cannot allocate tensor dimensions")?;
                dimensions
            } else {
                Vec::with_capacity(n_dimensions as usize)
            };
            for _ in 0..n_dimensions {
                let dim = read_length(reader, &magic)?;
                dimensions.push(usize::try_from(dim).with_context(|| {
                    format!("gguf: tensor '{tensor_name}' dimension {dim} does not fit usize")
                })?);
            }

            dimensions.reverse();
            let ggml_dtype = reader.read_u32::<LittleEndian>()?;
            let ggml_dtype = GgmlDType::from_u32(ggml_dtype)?;
            let offset = reader.read_u64::<LittleEndian>()?;
            tensor_infos.insert(
                tensor_name,
                TensorInfo {
                    shape: crate::Shape::from(dimensions),
                    offset,
                    ggml_dtype,
                },
            );
        }
        let position = reader.stream_position()?;
        let alignment = match metadata.get("general.alignment") {
            Some(Value::U8(v)) => *v as u64,
            Some(Value::U16(v)) => *v as u64,
            Some(Value::U32(v)) => *v as u64,
            Some(Value::I8(v)) if *v >= 0 => *v as u64,
            Some(Value::I16(v)) if *v >= 0 => *v as u64,
            Some(Value::I32(v)) if *v >= 0 => *v as u64,
            None => DEFAULT_ALIGNMENT,
            Some(value) => {
                crate::bail!(
                    "gguf: general.alignment must be a positive 8-, 16- or 32-bit integer, got type {:?}",
                    value.value_type()
                )
            }
        };
        if alignment == 0 {
            crate::bail!("gguf: general.alignment must not be zero")
        }
        let tensor_data_offset = position
            .checked_next_multiple_of(alignment)
            .context("gguf: aligned tensor data offset overflows u64")?;
        // No global data-offset/EOF check: an empty inventory has no spans.
        // In particular, seeking over optional final padding needs no data read.
        for (name, info) in &tensor_infos {
            info.byte_range(tensor_data_offset, file_size)
                .map_err(|err| err.context(format!("gguf: tensor '{name}'")))?;
        }
        Ok(Self {
            magic,
            metadata,
            tensor_infos,
            tensor_data_offset,
        })
    }

    pub fn tensor<R: std::io::Seek + std::io::Read>(
        &self,
        reader: &mut R,
        name: &str,
        device: &Device,
    ) -> Result<QTensor> {
        let tensor_info = match self.tensor_infos.get(name) {
            Some(tensor_info) => tensor_info,
            None => crate::bail!("cannot find tensor info for {name}"),
        };
        tensor_info.read(reader, self.tensor_data_offset, device)
    }
}

fn write_string<W: std::io::Write>(w: &mut W, str: &str) -> Result<()> {
    let bytes = str.as_bytes();
    w.write_u64::<LittleEndian>(bytes.len() as u64)?;
    w.write_all(bytes)?;
    Ok(())
}

pub fn write<W: std::io::Seek + std::io::Write>(
    w: &mut W,
    metadata: &[(&str, &Value)],
    tensors: &[(&str, &QTensor)],
) -> Result<()> {
    w.write_u32::<LittleEndian>(0x46554747)?;
    w.write_u32::<LittleEndian>(2)?; // version 2.
    w.write_u64::<LittleEndian>(tensors.len() as u64)?;
    w.write_u64::<LittleEndian>(metadata.len() as u64)?;
    for (name, value) in metadata.iter() {
        write_string(w, name)?;
        w.write_u32::<LittleEndian>(value.value_type().to_u32())?;
        value.write(w)?;
    }
    let mut offset = 0usize;
    let mut offsets = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors.iter() {
        write_string(w, name)?;
        let dims = tensor.shape().dims();
        w.write_u32::<LittleEndian>(dims.len() as u32)?;
        for &dim in dims.iter().rev() {
            w.write_u64::<LittleEndian>(dim as u64)?;
        }
        w.write_u32::<LittleEndian>(tensor.dtype().to_u32())?;
        w.write_u64::<LittleEndian>(offset as u64)?;
        offsets.push(offset);
        let size_in_bytes = tensor.storage_size_in_bytes();
        let padding = 31 - (31 + size_in_bytes) % 32;
        offset += size_in_bytes + padding;
    }
    let pos = w.stream_position()? as usize;
    let padding = 31 - (31 + pos) % 32;
    w.write_all(&vec![0u8; padding])?;
    let tensor_start_pos = w.stream_position()? as usize;
    for (offset, (_name, tensor)) in offsets.iter().zip(tensors.iter()) {
        let pos = w.stream_position()? as usize;
        if tensor_start_pos + offset != pos {
            crate::bail!(
                "internal error, unexpected current position {tensor_start_pos} {offset} {pos}"
            )
        }
        let data = tensor.data()?;
        let size_in_bytes = data.len();
        w.write_all(&data)?;
        let padding = 31 - (31 + size_in_bytes) % 32;
        w.write_all(&vec![0u8; padding])?;
    }
    Ok(())
}
