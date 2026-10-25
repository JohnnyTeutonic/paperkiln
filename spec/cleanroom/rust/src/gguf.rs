//! A GGUF reader (the container format of llama.cpp: header, metadata, tensor table, aligned data).
//! IR_SPEC does not describe GGUF itself; this follows llama.cpp's public description of version 3.

use crate::quant;
use crate::tensor::Tensor;

#[derive(Clone, Debug)]
pub enum Val {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(Vec<u8>),
    Arr(Vec<Val>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Val {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Val::U8(v) => Some(v as u64),
            Val::U16(v) => Some(v as u64),
            Val::U32(v) => Some(v as u64),
            Val::U64(v) => Some(v),
            Val::I8(v) if v >= 0 => Some(v as u64),
            Val::I16(v) if v >= 0 => Some(v as u64),
            Val::I32(v) if v >= 0 => Some(v as u64),
            Val::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<String> {
        match self {
            Val::Str(b) => Some(String::from_utf8_lossy(b).into_owned()),
            _ => None,
        }
    }
    pub fn is_u32(&self) -> bool {
        matches!(self, Val::U32(_))
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// ggml ne: ne[0] is the fastest-varying dimension.
    pub ne: Vec<u64>,
    pub ty: u32,
    pub offset: u64,
}

pub struct Gguf {
    pub kv: Vec<(String, Val)>,
    pub tensors: Vec<TensorInfo>,
    pub data: Vec<u8>,
    pub data_start: usize,
}

struct R<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> R<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.i + n > self.b.len() {
            return Err("GGUF truncated".into());
        }
        let s = &self.b[self.i..self.i + n];
        self.i += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn u64(&mut self) -> Result<u64, String> {
        let s = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        Ok(u64::from_le_bytes(a))
    }
    fn string(&mut self) -> Result<Vec<u8>, String> {
        let n = self.u64()? as usize;
        Ok(self.take(n)?.to_vec())
    }
    fn val(&mut self, t: u32) -> Result<Val, String> {
        Ok(match t {
            0 => Val::U8(self.take(1)?[0]),
            1 => Val::I8(self.take(1)?[0] as i8),
            2 => {
                let s = self.take(2)?;
                Val::U16(u16::from_le_bytes([s[0], s[1]]))
            }
            3 => {
                let s = self.take(2)?;
                Val::I16(i16::from_le_bytes([s[0], s[1]]))
            }
            4 => Val::U32(self.u32()?),
            5 => Val::I32(self.u32()? as i32),
            6 => Val::F32(f32::from_bits(self.u32()?)),
            7 => Val::Bool(self.take(1)?[0] != 0),
            8 => Val::Str(self.string()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.val(et)?);
                }
                Val::Arr(v)
            }
            10 => Val::U64(self.u64()?),
            11 => Val::I64(self.u64()? as i64),
            12 => Val::F64(f64::from_bits(self.u64()?)),
            _ => return Err(format!("GGUF metadata type {} unknown", t)),
        })
    }
}

impl Gguf {
    pub fn open(path: &str) -> Result<Gguf, String> {
        let data = std::fs::read(path).map_err(|e| format!("cannot read {}: {}", path, e))?;
        let mut r = R { b: &data, i: 0 };
        if r.take(4)? != b"GGUF" {
            return Err(format!("{} is not a GGUF file", path));
        }
        let ver = r.u32()?;
        if !(2..=3).contains(&ver) {
            return Err(format!("GGUF version {} not read", ver));
        }
        let nt = r.u64()? as usize;
        let nkv = r.u64()? as usize;
        let mut kv = Vec::new();
        for _ in 0..nkv {
            let k = String::from_utf8_lossy(&r.string()?).into_owned();
            let t = r.u32()?;
            let v = r.val(t)?;
            kv.push((k, v));
        }
        let mut tensors = Vec::new();
        for _ in 0..nt {
            let name = String::from_utf8_lossy(&r.string()?).into_owned();
            let nd = r.u32()? as usize;
            let mut ne = Vec::new();
            for _ in 0..nd {
                ne.push(r.u64()?);
            }
            let ty = r.u32()?;
            let offset = r.u64()?;
            tensors.push(TensorInfo { name, ne, ty, offset });
        }
        let align = kv.iter().find(|(k, _)| k == "general.alignment").and_then(|(_, v)| v.as_u64()).unwrap_or(32) as usize;
        let data_start = r.i.div_ceil(align) * align;
        Ok(Gguf { kv, tensors, data, data_start })
    }
    pub fn get(&self, key: &str) -> Option<&Val> {
        self.kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn get_str(&self, key: &str) -> Option<String> {
        self.get(key).and_then(|v| v.as_str())
    }
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
    /// The tensor's f32 values, row-major with shape [ne_{n-1}, ..., ne_0].
    pub fn load(&self, name: &str) -> Result<Tensor, String> {
        let ti = self.info(name).ok_or(format!("tensor {} is not in the file", name))?;
        let n: usize = ti.ne.iter().product::<u64>() as usize;
        let (be, bb) = quant::block_size(ti.ty).ok_or(format!("tensor {}: storage type {} not read", name, ti.ty))?;
        let nbytes = n / be * bb;
        let start = self.data_start + ti.offset as usize;
        if start + nbytes > self.data.len() {
            return Err(format!("tensor {} lies outside the file", name));
        }
        let vals = quant::dequantize(ti.ty, &self.data[start..start + nbytes], n)?;
        let shape: Vec<usize> = ti.ne.iter().rev().map(|&x| x as usize).collect();
        Ok(Tensor::new(shape, vals))
    }
}
