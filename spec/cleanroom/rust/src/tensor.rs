//! Dense f32 tensors, row-major, and the suite's hex encoding (IR_SPEC 11.2).

use crate::json::Json;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        Tensor { shape, data }
    }
    pub fn zeros(shape: Vec<usize>) -> Tensor {
        let n = shape.iter().product();
        Tensor { shape, data: vec![0.0; n] }
    }
    pub fn numel(&self) -> usize {
        self.data.len()
    }
    /// Rows of a [T, w] value (a 1-D tensor is one row).
    pub fn rows(&self) -> usize {
        if self.shape.len() >= 2 {
            self.shape[0]
        } else {
            1
        }
    }
    pub fn cols(&self) -> usize {
        if self.shape.is_empty() {
            1
        } else if self.shape.len() == 1 {
            self.shape[0]
        } else {
            self.shape[1..].iter().product()
        }
    }
    pub fn row(&self, t: usize) -> &[f32] {
        let w = self.cols();
        &self.data[t * w..(t + 1) * w]
    }
    pub fn mat(rows: usize, cols: usize, data: Vec<f32>) -> Tensor {
        Tensor::new(vec![rows, cols], data)
    }
    pub fn gather_rows(&self, idx: &[usize]) -> Tensor {
        let w = self.cols();
        let mut d = Vec::with_capacity(idx.len() * w);
        for &t in idx {
            d.extend_from_slice(self.row(t));
        }
        Tensor::mat(idx.len(), w, d)
    }
}

pub fn decode_hex_f32(h: &str) -> Result<Vec<f32>, String> {
    if h.len() % 8 != 0 {
        return Err("tensor hex is not a multiple of 8 digits".into());
    }
    let mut out = Vec::with_capacity(h.len() / 8);
    for i in (0..h.len()).step_by(8) {
        let bits = u32::from_str_radix(&h[i..i + 8], 16).map_err(|_| "bad tensor hex".to_string())?;
        out.push(f32::from_bits(bits));
    }
    Ok(out)
}

pub fn encode_hex_f32(d: &[f32]) -> String {
    let mut s = String::with_capacity(d.len() * 8);
    for x in d {
        s.push_str(&format!("{:08x}", x.to_bits()));
    }
    s
}

pub fn decode_hex_bytes(h: &str) -> Result<Vec<u8>, String> {
    if h.len() % 2 != 0 {
        return Err("odd hex length".into());
    }
    (0..h.len()).step_by(2).map(|i| u8::from_str_radix(&h[i..i + 2], 16).map_err(|_| "bad hex".to_string())).collect()
}

pub fn tensor_from_json(j: &Json) -> Result<Tensor, String> {
    let shape: Vec<usize> = j
        .get("shape")
        .and_then(|s| s.as_arr())
        .ok_or("tensor without shape")?
        .iter()
        .map(|x| x.as_i64().map(|v| v as usize).ok_or("bad shape"))
        .collect::<Result<_, _>>()?;
    let data = decode_hex_f32(j.get("f32").and_then(|s| s.as_str()).ok_or("tensor without f32")?)?;
    if shape.iter().product::<usize>() != data.len() {
        return Err(format!("tensor shape {:?} does not hold {} values", shape, data.len()));
    }
    Ok(Tensor { shape, data })
}

pub fn tensor_to_json(t: &Tensor) -> Json {
    Json::obj(vec![
        ("shape", Json::Arr(t.shape.iter().map(|&s| Json::int(s as i64)).collect())),
        ("f32", Json::Str(encode_hex_f32(&t.data))),
    ])
}
