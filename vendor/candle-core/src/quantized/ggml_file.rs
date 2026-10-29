//! Support for the GGML file format.

use super::{k_quants, GgmlDType, QStorage};
use crate::{Context, Device, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;

// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.h#L37
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Magic {
    Ggjt,
    Ggla,
    Ggmf,
    Ggml,
    Ggsn,
}

impl TryFrom<u32> for Magic {
    type Error = crate::Error;
    fn try_from(value: u32) -> Result<Self> {
        let magic = match value {
            0x67676a74 => Self::Ggjt,
            0x67676c61 => Self::Ggla,
            0x67676d66 => Self::Ggmf,
            0x67676d6c => Self::Ggml,
            0x6767736e => Self::Ggsn,
            _ => crate::bail!("unknown magic {value:08x}"),
        };
        Ok(magic)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgmlUnversioned,
    GgmfV1,
    GgjtV1,
    GgjtV2,
    GgjtV3,
}

impl VersionedMagic {
    fn read<R: std::io::Read>(reader: &mut R) -> Result<Self> {
        let magic = reader.read_u32::<LittleEndian>()?;
        let magic = Magic::try_from(magic)?;
        if magic == Magic::Ggml {
            return Ok(Self::GgmlUnversioned);
        }
        let version = reader.read_u32::<LittleEndian>()?;
        let versioned_magic = match (magic, version) {
            (Magic::Ggmf, 1) => Self::GgmfV1,
            (Magic::Ggjt, 1) => Self::GgjtV1,
            (Magic::Ggjt, 2) => Self::GgjtV2,
            (Magic::Ggjt, 3) => Self::GgjtV3,
            _ => crate::bail!("ggml: unsupported magic/version {magic:?}/{version}"),
        };
        Ok(versioned_magic)
    }

    fn align32(&self) -> bool {
        match self {
            Self::GgmlUnversioned | Self::GgmfV1 => false,
            Self::GgjtV1 | Self::GgjtV2 | Self::GgjtV3 => true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HParams {
    pub n_vocab: u32,
    pub n_embd: u32,
    pub n_mult: u32,
    pub n_head: u32,
    pub n_layer: u32,
    pub n_rot: u32,
    pub ftype: u32,
}

impl HParams {
    fn read<R: std::io::Read>(reader: &mut R) -> Result<Self> {
        let n_vocab = reader.read_u32::<LittleEndian>()?;
        let n_embd = reader.read_u32::<LittleEndian>()?;
        let n_mult = reader.read_u32::<LittleEndian>()?;
        let n_head = reader.read_u32::<LittleEndian>()?;
        let n_layer = reader.read_u32::<LittleEndian>()?;
        let n_rot = reader.read_u32::<LittleEndian>()?;
        let ftype = reader.read_u32::<LittleEndian>()?;
        Ok(Self {
            n_vocab,
            n_embd,
            n_mult,
            n_head,
            n_layer,
            n_rot,
            ftype,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Vocab {
    pub token_score_pairs: Vec<(Vec<u8>, f32)>,
}

impl Vocab {
    fn read<R: std::io::Read>(reader: &mut R, n_vocab: usize) -> Result<Self> {
        // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.cpp#L556
        let mut token_score_pairs = Vec::with_capacity(n_vocab);
        for _index in 0..n_vocab {
            let len = reader.read_u32::<LittleEndian>()? as usize;
            let mut word = vec![0u8; len];
            reader.read_exact(&mut word)?;
            let score = reader.read_f32::<LittleEndian>()?;
            token_score_pairs.push((word, score))
        }
        Ok(Self { token_score_pairs })
    }
}

fn from_data<T: super::GgmlType + Send + Sync + 'static>(
    data: Vec<T>,
    dims: Vec<usize>,
    device: &Device,
) -> Result<super::QTensor> {
    let data: QStorage = match device {
        Device::Cpu => QStorage::Cpu(Box::new(data)),
        Device::Metal(metal) => super::metal::load_quantized(metal, &data)?,
        Device::Cuda(cuda) => {
            #[cfg(feature = "cuda")]
            {
                use crate::cuda_backend::cudarc::driver::sys::CUstreamCaptureStatus;

                let stream = cuda.cuda_stream();
                let status = stream
                    .capture_status()
                    .context("ggml: checking CUDA tensor upload capture status")?;
                if status != CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
                    crate::bail!(
                        "ggml: tensor materialization during CUDA graph capture is unsupported"
                    )
                }
                // The upload helper only borrows this private allocation. Keep it
                // alive through completion, even if the upload itself returns an
                // error. Captured copies cannot use this finite source lifetime.
                let uploaded = super::cuda::load_quantized(cuda, &data);
                if let Err(error) = stream
                    .synchronize()
                    .context("ggml: confirming CUDA tensor upload completion")
                {
                    // An unconfirmed copy may still read these bytes. Retain the
                    // allocation instead of freeing a potentially live source.
                    std::mem::forget(data);
                    let context = format!(
                        "ggml: host tensor buffer retained after unconfirmed CUDA upload ({error})"
                    );
                    return Err(uploaded.err().unwrap_or(error)).context(context);
                }
                uploaded?
            }
            #[cfg(not(feature = "cuda"))]
            super::cuda::load_quantized(cuda, &data)?
        }
    };
    super::QTensor::new(data, dims)
}

fn tensor_size_in_bytes(ggml_dtype: GgmlDType, dims: &[usize]) -> Result<usize> {
    let Some(&last_dim) = dims.last() else {
        crate::bail!("ggml: scalar tensor cannot be quantized")
    };
    // Nonzero dimensions also ensure every reverse contiguous-stride product
    // is bounded by the checked total, without a zero masking overflow.
    if dims.contains(&0) {
        crate::bail!("ggml: empty tensor cannot be materialized")
    }
    let tensor_elems = dims.iter().try_fold(1usize, |n, &dim| {
        n.checked_mul(dim)
            .context("ggml: tensor element count overflows usize")
    })?;
    let block_size = ggml_dtype.block_size();
    if !last_dim.is_multiple_of(block_size) {
        crate::bail!(
            "ggml: tensor last dimension {last_dim} is not divisible by block size {block_size}"
        )
    }
    (tensor_elems / block_size)
        .checked_mul(ggml_dtype.type_size())
        .context("ggml: tensor byte length overflows usize")
}

/// Creates a [Tensor] from a raw GGML tensor.
///
/// Scalars and zero-element shapes are rejected. Only the required byte prefix
/// is consumed; trailing bytes are ignored. Values retain the existing native
/// endian interpretation of the GGML block layouts.
pub fn qtensor_from_ggml(
    ggml_dtype: GgmlDType,
    raw_data: &[u8],
    dims: Vec<usize>,
    device: &Device,
) -> Result<super::QTensor> {
    let size_in_bytes = tensor_size_in_bytes(ggml_dtype, &dims)?;
    let raw_data = raw_data
        .get(..size_in_bytes)
        .context("ggml: raw tensor buffer is shorter than its shape requires")?;

    // Keep the bit-validity proof local to this concrete dispatch, rather than
    // trusting arbitrary implementations of the public, safe GgmlType trait.
    macro_rules! decode {
        ($ty:ty) => {{
            let mut data = Vec::<$ty>::new();
            data.try_reserve_exact(raw_data.len() / std::mem::size_of::<$ty>())
                .context("ggml: cannot allocate typed tensor buffer")?;
            for block in raw_data.chunks_exact(std::mem::size_of::<$ty>()) {
                // SAFETY: Each chunk contains exactly size_of::<$ty>() readable
                // bytes, alive and immutable during the read. read_unaligned
                // requires no source alignment. The concrete types below are
                // f32, transparent u16 half floats, or repr(C) blocks containing
                // only those floats and integer arrays, with no padding (see
                // k_quants' size assertions). All bit patterns are valid, with
                // no references, ownership or Drop state to duplicate. The Vec
                // owns the resulting values in properly aligned storage.
                data.push(unsafe { block.as_ptr().cast::<$ty>().read_unaligned() });
            }
            from_data(data, dims, device)
        }};
    }

    match ggml_dtype {
        GgmlDType::F32 => decode!(f32),
        GgmlDType::F16 => decode!(half::f16),
        GgmlDType::BF16 => decode!(half::bf16),
        GgmlDType::Q4_0 => decode!(k_quants::BlockQ4_0),
        GgmlDType::Q4_1 => decode!(k_quants::BlockQ4_1),
        GgmlDType::Q5_0 => decode!(k_quants::BlockQ5_0),
        GgmlDType::Q5_1 => decode!(k_quants::BlockQ5_1),
        GgmlDType::Q8_0 => decode!(k_quants::BlockQ8_0),
        GgmlDType::Q2K => decode!(k_quants::BlockQ2K),
        GgmlDType::Q3K => decode!(k_quants::BlockQ3K),
        GgmlDType::Q4K => decode!(k_quants::BlockQ4K),
        GgmlDType::Q5K => decode!(k_quants::BlockQ5K),
        GgmlDType::Q6K => decode!(k_quants::BlockQ6K),
        _ => crate::bail!("quantized type {ggml_dtype:?} is not supported yet"),
    }
}

fn read_one_tensor<R: std::io::Seek + std::io::Read>(
    reader: &mut R,
    magic: VersionedMagic,
    device: &Device,
) -> Result<(String, super::QTensor)> {
    let n_dims = reader.read_u32::<LittleEndian>()?;
    let name_len = reader.read_u32::<LittleEndian>()?;
    let ggml_dtype = reader.read_u32::<LittleEndian>()?;
    let ggml_dtype = GgmlDType::from_u32(ggml_dtype)?;
    let mut dims = vec![0u32; n_dims as usize];
    reader.read_u32_into::<LittleEndian>(&mut dims)?;
    // The dimensions are stored in reverse order, see for example:
    // https://github.com/ggerganov/llama.cpp/blob/b5ffb2849d23afe73647f68eec7b68187af09be6/convert.py#L969
    dims.reverse();
    let mut name = vec![0u8; name_len as usize];
    reader.read_exact(&mut name)?;
    let name = String::from_utf8_lossy(&name).into_owned();

    if magic.align32() {
        let pos = reader.stream_position()?;
        reader.seek(std::io::SeekFrom::Current(((32 - pos % 32) % 32) as i64))?;
    }
    let dims = dims.iter().map(|&u| u as usize).collect::<Vec<_>>();
    let size_in_bytes = tensor_size_in_bytes(ggml_dtype, &dims)?;
    // TODO: Mmap version to avoid copying the data around?
    let mut raw_data = Vec::new();
    raw_data
        .try_reserve_exact(size_in_bytes)
        .context("ggml: cannot allocate tensor read buffer")?;
    raw_data.resize(size_in_bytes, 0u8);
    reader.read_exact(&mut raw_data)?;
    match qtensor_from_ggml(ggml_dtype, &raw_data, dims, device) {
        Ok(tensor) => Ok((name, tensor)),
        Err(e) => crate::bail!("Error creating tensor {name}: {e}"),
    }
}

pub struct Content {
    pub magic: VersionedMagic,
    pub hparams: HParams,
    pub vocab: Vocab,
    pub tensors: HashMap<String, super::QTensor>,
    pub device: Device,
}

impl Content {
    pub fn read<R: std::io::Seek + std::io::Read>(
        reader: &mut R,
        device: &Device,
    ) -> Result<Content> {
        // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.cpp#L505
        let last_position = reader.seek(std::io::SeekFrom::End(0))?;
        reader.seek(std::io::SeekFrom::Start(0))?;
        let magic = VersionedMagic::read(reader)?;
        let hparams = HParams::read(reader)?;
        let vocab = Vocab::read(reader, hparams.n_vocab as usize)?;
        let mut tensors = HashMap::new();

        while reader.stream_position()? != last_position {
            let (name, tensor) = read_one_tensor(reader, magic, device)?;
            tensors.insert(name, tensor);
        }
        let device = device.clone();
        Ok(Self {
            magic,
            hparams,
            vocab,
            tensors,
            device,
        })
    }

    pub fn remove(&mut self, name: &str) -> Result<super::QTensor> {
        match self.tensors.remove(name) {
            None => crate::bail!("cannot find tensor with name '{name}'"),
            Some(tensor) => Ok(tensor),
        }
    }
}
