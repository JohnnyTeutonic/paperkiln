//! The ops of IR_SPEC sections 5 to 7, each in the order of operations its text states.
//! Every reduction is sequential and ascending, so no value depends on how many rows a call runs.

use crate::expr;
use crate::json::Json;
use crate::model::{at, at_f32, at_i64, at_str, at_usize, ins, op, outs, Model};
use crate::tensor::Tensor;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub enum Val {
    T(Tensor),
    L(Vec<Tensor>),
    I(Vec<i64>),
}

pub type Env = HashMap<String, Val>;

/// The k/v rows one application's instance of a kv slot holds, in ascending position.
#[derive(Clone, Debug, Default)]
pub struct KvInst {
    pub pos: Vec<usize>,
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, Default)]
pub struct Slots {
    pub kv: HashMap<(String, usize), KvInst>,
    /// recurrent slot tensors, in the slot's declaration order
    pub rec: HashMap<(String, usize), Vec<Tensor>>,
}

/// turn_state codes (IR_SPEC 6.8.2) by absolute position.
#[derive(Clone, Debug, Default)]
pub struct Codes {
    pub c: Vec<bool>,
    pub s: Vec<usize>,
    /// complete blocks after the last position
    pub total: usize,
}

impl Codes {
    pub fn compute(ids: &[i64], vocab: usize, k: usize) -> Codes {
        let mut c = Vec::with_capacity(ids.len());
        let mut s = Vec::with_capacity(ids.len());
        let mut done = 0usize;
        let mut run = 0usize;
        for &id in ids {
            s.push(done);
            let is_state = id >= vocab as i64;
            c.push(is_state);
            if !is_state || k == 0 {
                run = 0;
                continue;
            }
            let j = (id - vocab as i64) as usize;
            if j == 0 {
                run = 1;
            } else if j == run {
                run += 1;
            } else {
                run = 0;
            }
            if run == k {
                done += 1;
                run = 0;
            }
        }
        Codes { c, s, total: done }
    }
    /// complete blocks among the first n positions (the segment position n belongs to)
    pub fn blocks_before(&self, n: usize) -> usize {
        if n < self.s.len() {
            self.s[n]
        } else {
            self.total
        }
    }
    /// the first position of segment m: the position after block m - 1's last, or 0
    pub fn segment_start(&self, m: usize) -> usize {
        if m == 0 {
            return 0;
        }
        (0..=self.s.len()).find(|&p| self.blocks_before(p) >= m).unwrap_or(self.s.len())
    }
}

/// The rows a call runs: their absolute positions and ids.
#[derive(Clone, Debug)]
pub struct Rows {
    pub pos: Vec<usize>,
}

pub struct Ctx<'a> {
    pub model: &'a Model,
    /// None outside the block
    pub app: Option<usize>,
    pub rows: &'a Rows,
    pub codes: Option<&'a Codes>,
    pub section: u8,
    pub node_index: usize,
}

fn tensor<'a>(env: &'a Env, name: &str) -> Result<&'a Tensor, String> {
    match env.get(name) {
        Some(Val::T(t)) => Ok(t),
        Some(_) => Err(format!("value {} is not a tensor", name)),
        None => Err(format!("value {} has not been written", name)),
    }
}

fn ints<'a>(env: &'a Env, name: &str) -> Result<&'a Vec<i64>, String> {
    match env.get(name) {
        Some(Val::I(v)) => Ok(v),
        _ => Err(format!("value {} is not an integer list", name)),
    }
}

fn list<'a>(env: &'a Env, name: &str) -> Result<&'a Vec<Tensor>, String> {
    match env.get(name) {
        Some(Val::L(v)) => Ok(v),
        Some(Val::T(_)) => Err(format!("value {} is a tensor, not a list", name)),
        _ => Err(format!("value {} has not been written", name)),
    }
}

fn out1(node: &Json) -> Result<String, String> {
    outs(node).into_iter().next().ok_or(format!("{}: no out", op(node)))
}

fn put(env: &mut Env, node: &Json, t: Tensor) -> Result<(), String> {
    env.insert(out1(node)?, Val::T(t));
    Ok(())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// x W^T + bias: W [out, in] row-major; each dot product summed in ascending i from +0.
pub fn linear(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<Tensor, String> {
    let (t, n) = (x.rows(), x.cols());
    let o = if w.shape.len() >= 2 { w.shape[0] } else { 1 };
    if w.numel() != o * n {
        return Err(format!("linear: weight {:?} does not take inputs of width {}", w.shape, n));
    }
    if let Some(b) = bias {
        if b.numel() != o {
            return Err(format!("linear: bias of {} values for {} outputs", b.numel(), o));
        }
    }
    let mut y = Vec::with_capacity(t * o);
    for r in 0..t {
        let xr = x.row(r);
        for j in 0..o {
            let mut s = dot(xr, &w.data[j * n..(j + 1) * n]);
            if let Some(b) = bias {
                s += b.data[j];
            }
            y.push(s);
        }
    }
    Ok(Tensor::mat(t, o, y))
}

fn softmax_inplace(l: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &x in l.iter() {
        if x > m {
            m = x;
        }
    }
    let mut z = 0.0f32;
    for x in l.iter_mut() {
        *x = (*x - m).exp();
        z += *x;
    }
    for x in l.iter_mut() {
        *x /= z;
    }
}

fn gelu_tanh(x: f32) -> f32 {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    let x3 = x * x * x;
    let inner = c * (x + 0.044715f32 * x3);
    0.5f32 * x * (1.0f32 + inner.tanh())
}

fn gelu_erf(x: f32) -> f32 {
    0.5f32 * x * (1.0f32 + expr::erf32(x / std::f32::consts::SQRT_2))
}

fn same_shape(a: &Tensor, b: &Tensor, o: &str) -> Result<(), String> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(format!("{}: inputs of shapes {:?} and {:?}", o, a.shape, b.shape));
    }
    Ok(())
}

fn phi(x: f32) -> f32 {
    if x > 0.0 {
        x + 1.0
    } else {
        (x.exp() - 1.0) + 1.0
    }
}

/// Runs one node (every op except depth_route, which the block runner handles).
pub fn exec(ctx: &Ctx, node: &Json, env: &mut Env, slots: &mut Slots) -> Result<(), String> {
    let model = ctx.model;
    let app = ctx.app;
    let a = app.unwrap_or(0);
    let i = ins(node);
    let o = op(node);
    let input = |k: usize| -> Result<&Tensor, String> { tensor(env, i.get(k).ok_or(format!("{}: missing input {}", o, k))?) };
    match o {
        "embedding" => {
            let ids = ints(env, &i[0])?.clone();
            let w = model.wa(node, "weight", app)?;
            let sw = model.wa_opt(node, "state_weight", app)?;
            let d = w.cols();
            let vocab = if model.single_node { w.rows() } else { model.vocab };
            let mut out = Vec::with_capacity(ids.len() * d);
            for &id in &ids {
                if id < 0 {
                    return Err(format!("embedding: negative id {}", id));
                }
                let id = id as usize;
                if id < vocab {
                    if id >= w.rows() {
                        return Err(format!("embedding: id {} beyond the table's {} rows", id, w.rows()));
                    }
                    out.extend_from_slice(w.row(id));
                } else {
                    match sw {
                        Some(s) if id - vocab < s.rows() => {
                            if s.cols() != d {
                                return Err("embedding: state_weight width differs from the table's".into());
                            }
                            out.extend_from_slice(s.row(id - vocab));
                        }
                        Some(s) => return Err(format!("embedding: id {} >= vocab + k = {} (IR_SPEC 6.8.1)", id, vocab + s.rows())),
                        None => return Err(format!("embedding: id {} >= vocab {} (IR_SPEC 5.1)", id, vocab)),
                    }
                }
            }
            put(env, node, Tensor::mat(ids.len(), d, out))
        }
        "pos_embedding" => {
            let x = input(0)?;
            let pos = ints(env, &i[1])?;
            let w = model.wa(node, "weight", app)?;
            if w.cols() != x.cols() || pos.len() != x.rows() {
                return Err("pos_embedding: shapes do not match".into());
            }
            let mut y = x.data.clone();
            let d = x.cols();
            for (t, &p) in pos.iter().enumerate() {
                if p < 0 || p as usize >= w.rows() {
                    return Err(format!("pos_embedding: position {} >= ctx {} (IR_SPEC 5.2)", p, w.rows()));
                }
                let wr = w.row(p as usize);
                for j in 0..d {
                    y[t * d + j] += wr[j];
                }
            }
            put(env, node, Tensor::mat(x.rows(), d, y))
        }
        "linear" => {
            let x = input(0)?;
            let wn = model.tname(node, node.get("weight").and_then(|v| v.as_str()).unwrap_or(""), app);
            let w = match model.weights.get(&wn) {
                Some(w) => w,
                None => match node.get("tied_to").and_then(|v| v.as_str()) {
                    Some(t) => model.w(&model.tname(node, t, app))?,
                    None => return Err(format!("linear: tensor {} is not a tensor of the file", wn)),
                },
            };
            let b = model.wa_opt(node, "bias", app)?;
            let y = linear(x, w, b)?;
            put(env, node, y)
        }
        "scale" => {
            let x = input(0)?;
            let w = model.wa(node, "weight", app)?;
            let b = model.wa_opt(node, "bias", app)?;
            let d = x.cols();
            if w.numel() != d || b.map(|b| b.numel() != d).unwrap_or(false) {
                return Err(format!("scale: weight or bias length is not the width {} (IR_SPEC 5.4)", d));
            }
            let mut y = Vec::with_capacity(x.numel());
            for t in 0..x.rows() {
                let r = x.row(t);
                for j in 0..d {
                    let p = r[j] * w.data[j];
                    y.push(match b {
                        Some(b) => p + b.data[j],
                        None => p,
                    });
                }
            }
            put(env, node, Tensor::mat(x.rows(), d, y))
        }
        "split" => {
            let x = input(0)?.clone();
            let sizes: Vec<usize> =
                node.get("sizes").and_then(|s| s.as_arr()).ok_or("split: no sizes")?.iter().map(|v| v.as_i64().unwrap_or(0) as usize).collect();
            let names = outs(node);
            if sizes.iter().sum::<usize>() != x.cols() || names.len() != sizes.len() {
                return Err(format!("split: sizes {:?} do not cover width {}", sizes, x.cols()));
            }
            let mut start = 0;
            for (k, &s) in sizes.iter().enumerate() {
                let mut d = Vec::with_capacity(x.rows() * s);
                for t in 0..x.rows() {
                    d.extend_from_slice(&x.row(t)[start..start + s]);
                }
                env.insert(names[k].clone(), Val::T(Tensor::mat(x.rows(), s, d)));
                start += s;
            }
            Ok(())
        }
        "rmsnorm" => {
            let x = input(0)?;
            let w = model.wa(node, "weight", app)?;
            let eps = at_f32(node, "eps", a)?;
            let width = x.cols();
            let g = match at(node, "group", a) {
                Some(v) => v.as_i64().unwrap_or(0) as usize,
                None => width,
            };
            if g == 0 || width % g != 0 || w.numel() != g {
                return Err(format!("rmsnorm: group {} and weight of {} for width {}", g, w.numel(), width));
            }
            let off = match at(node, "weight_offset", a) {
                Some(v) => v.as_f64().unwrap_or(0.0) as f32,
                None => 0.0,
            };
            let wv: Vec<f32> = w.data.iter().map(|&v| if off != 0.0 { v + off } else { v }).collect();
            let mut y = Vec::with_capacity(x.numel());
            for t in 0..x.rows() {
                let r = x.row(t);
                for gs in (0..width).step_by(g) {
                    let mut ss = 0.0f32;
                    for j in gs..gs + g {
                        ss += r[j] * r[j];
                    }
                    let mean = ss / g as f32;
                    let den = (mean + eps).sqrt();
                    for j in gs..gs + g {
                        y.push((r[j] / den) * wv[j - gs]);
                    }
                }
            }
            put(env, node, Tensor::mat(x.rows(), width, y))
        }
        "layernorm" => {
            let x = input(0)?;
            let w = model.wa(node, "weight", app)?;
            let b = model.wa(node, "bias", app)?;
            let eps = at_f32(node, "eps", a)?;
            let d = x.cols();
            if w.numel() != d || b.numel() != d {
                return Err("layernorm: weight or bias length is not the width".into());
            }
            let mut y = Vec::with_capacity(x.numel());
            for t in 0..x.rows() {
                let r = x.row(t);
                let mut s = 0.0f32;
                for &v in r {
                    s += v;
                }
                let mean = s / d as f32;
                let mut ss = 0.0f32;
                for &v in r {
                    let dv = v - mean;
                    ss += dv * dv;
                }
                let var = ss / d as f32;
                let den = (var + eps).sqrt();
                for j in 0..d {
                    y.push(((r[j] - mean) / den) * w.data[j] + b.data[j]);
                }
            }
            put(env, node, Tensor::mat(x.rows(), d, y))
        }
        "gelu" => {
            let x = input(0)?;
            let f: fn(f32) -> f32 = match at_str(node, "approx", a) {
                Some("tanh") => gelu_tanh,
                Some("none") => gelu_erf,
                v => return Err(format!("gelu: approx {:?} is not tanh or none", v)),
            };
            put(env, node, Tensor::new(x.shape.clone(), x.data.iter().map(|&v| f(v)).collect()))
        }
        "silu" | "sigmoid" | "relu" => {
            let x = input(0)?;
            let f: fn(f32) -> f32 = match o {
                "silu" => |v| v * expr::sigmoid(v),
                "sigmoid" => expr::sigmoid,
                _ => expr::relu,
            };
            put(env, node, Tensor::new(x.shape.clone(), x.data.iter().map(|&v| f(v)).collect()))
        }
        "add" | "mul" | "sub" => {
            let x = input(0)?;
            let y = input(1)?;
            same_shape(x, y, o)?;
            let d: Vec<f32> = x
                .data
                .iter()
                .zip(&y.data)
                .map(|(&p, &q)| match o {
                    "add" => p + q,
                    "mul" => p * q,
                    _ => p - q,
                })
                .collect();
            put(env, node, Tensor::new(x.shape.clone(), d))
        }
        "affine" => {
            let x = input(0)?;
            let s = at_f32(node, "scale", a)?;
            let sh = at_f32(node, "shift", a)?;
            put(env, node, Tensor::new(x.shape.clone(), x.data.iter().map(|&v| (v * s) + sh).collect()))
        }
        "mul_col" => {
            let x = input(0)?;
            let c = input(1)?;
            if c.cols() != 1 || c.rows() != x.rows() {
                return Err("mul_col: c is not [T, 1]".into());
            }
            let w = x.cols();
            let d: Vec<f32> = (0..x.numel()).map(|k| x.data[k] * c.data[k / w]).collect();
            put(env, node, Tensor::new(x.shape.clone(), d))
        }
        "rms_row" => {
            let x = input(0)?;
            let eps = at_f32(node, "eps", a)?;
            let w = x.cols();
            let d: Vec<f32> = (0..x.rows())
                .map(|t| {
                    let mut ss = 0.0f32;
                    for &v in x.row(t) {
                        ss += v * v;
                    }
                    (ss / w as f32 + eps).sqrt()
                })
                .collect();
            put(env, node, Tensor::mat(x.rows(), 1, d))
        }
        "rope" => {
            let q = input(0)?.clone();
            let k = input(1)?.clone();
            let pos = ints(env, &i[2])?.clone();
            let heads = at_usize(node, "heads", a)?;
            let hd = at_usize(node, "head_dim", a)?;
            let kvh = match at(node, "kv_heads", a) {
                Some(v) => v.as_i64().unwrap_or(0) as usize,
                None => heads,
            };
            let n_rot = match at(node, "n_rot", a) {
                Some(v) => v.as_i64().unwrap_or(0) as usize,
                None => hd,
            };
            let theta = at_f32(node, "theta", a)?;
            let fs = match at(node, "freq_scale", a) {
                Some(v) => v.as_f64().unwrap_or(1.0) as f32,
                None => 1.0,
            };
            let first = match at_str(node, "rope_heads", a) {
                Some("all") => false,
                Some("first") => true,
                v => return Err(format!("rope: rope_heads {:?}", v)),
            };
            let neox = match at_str(node, "pairing", a) {
                Some("interleaved") => false,
                Some("neox") => true,
                v => return Err(format!("rope: pairing {:?}", v)),
            };
            if q.cols() != heads * hd || k.cols() != kvh * hd || pos.len() != q.rows() || k.rows() != q.rows() {
                return Err(format!("rope: q {:?} / k {:?} do not fit heads {} kv_heads {} head_dim {}", q.shape, k.shape, heads, kvh, hd));
            }
            if n_rot == 0 || n_rot % 2 != 0 || n_rot > hd {
                return Err("rope: n_rot".into());
            }
            let half = n_rot / 2;
            let freqs: Vec<f32> = (0..half).map(|j| 1.0f32 / theta.powf((2 * j) as f32 / n_rot as f32)).collect();
            let rot = |x: &Tensor, nh: usize| -> Tensor {
                let mut d = x.data.clone();
                let w = x.cols();
                for t in 0..x.rows() {
                    let p = pos[t] as f32;
                    for h in 0..(if first { 1 } else { nh }) {
                        let base = t * w + h * hd;
                        for j in 0..half {
                            let ang = (p * fs) * freqs[j];
                            let (s, c) = (ang.sin(), ang.cos());
                            let (i0, i1) = if neox { (base + j, base + j + half) } else { (base + 2 * j, base + 2 * j + 1) };
                            let x0 = x.data[i0];
                            let x1 = x.data[i1];
                            d[i0] = x0 * c - x1 * s;
                            d[i1] = x0 * s + x1 * c;
                        }
                    }
                }
                Tensor::new(x.shape.clone(), d)
            };
            let qn = rot(&q, heads);
            let kn = rot(&k, kvh);
            let names = outs(node);
            if names.len() != 2 {
                return Err("rope: out is not two names".into());
            }
            env.insert(names[0].clone(), Val::T(qn));
            env.insert(names[1].clone(), Val::T(kn));
            Ok(())
        }
        "attention" => attention(ctx, node, env, slots),
        "moe_ffn" => {
            let x = input(0)?;
            let r = model.wa(node, "router", app)?;
            let ex = node.get("experts").ok_or("moe_ffn: no experts")?;
            let tn = |k: &str| -> Result<&Tensor, String> {
                let n = ex.get(k).and_then(|v| v.as_str()).ok_or(format!("moe_ffn: experts.{}", k))?;
                model.w(&model.tname(node, n, app))
            };
            let (g, u, dn) = (tn("gate")?, tn("up")?, tn("down")?);
            let e_n = at_usize(node, "n_experts", a)?;
            let top_k = at_usize(node, "top_k", a)?;
            let renorm = match at_str(node, "gate_norm", a) {
                Some("renorm") => true,
                Some("raw") => false,
                v => return Err(format!("moe_ffn: gate_norm {:?}", v)),
            };
            let d = x.cols();
            if g.shape.len() != 3 || g.shape[0] != e_n || g.shape[2] != d {
                return Err(format!("moe_ffn: gate stack {:?}", g.shape));
            }
            let ff = g.shape[1];
            let mut y = Vec::with_capacity(x.numel());
            for t in 0..x.rows() {
                let xr = x.row(t);
                let logits: Vec<f32> = (0..e_n).map(|e| dot(xr, &r.data[e * d..(e + 1) * d])).collect();
                // top-k, ties to the lower index
                let mut chosen: Vec<usize> = Vec::new();
                for _ in 0..top_k {
                    let mut best: Option<usize> = None;
                    for e in 0..e_n {
                        if chosen.contains(&e) {
                            continue;
                        }
                        if best.map(|b| logits[e] > logits[b]).unwrap_or(true) {
                            best = Some(e);
                        }
                    }
                    chosen.push(best.unwrap());
                }
                chosen.sort();
                let gates: Vec<f32> = if renorm {
                    let mut l: Vec<f32> = chosen.iter().map(|&e| logits[e]).collect();
                    softmax_inplace(&mut l);
                    l
                } else {
                    let mut p = logits.clone();
                    softmax_inplace(&mut p);
                    chosen.iter().map(|&e| p[e]).collect()
                };
                let mut acc = vec![0.0f32; d];
                for (ci, &e) in chosen.iter().enumerate() {
                    let gw = &g.data[e * ff * d..(e + 1) * ff * d];
                    let uw = &u.data[e * ff * d..(e + 1) * ff * d];
                    let dw = &dn.data[e * d * ff..(e + 1) * d * ff];
                    let hidden: Vec<f32> = (0..ff)
                        .map(|f| {
                            let gv = dot(xr, &gw[f * d..(f + 1) * d]);
                            let uv = dot(xr, &uw[f * d..(f + 1) * d]);
                            (gv * expr::sigmoid(gv)) * uv
                        })
                        .collect();
                    for j in 0..d {
                        let ev = dot(&hidden, &dw[j * ff..(j + 1) * ff]);
                        acc[j] += gates[ci] * ev;
                    }
                }
                y.extend_from_slice(&acc);
            }
            put(env, node, Tensor::mat(x.rows(), d, y))
        }
        "list" => {
            let mut l = Vec::new();
            for n in &i {
                l.push(tensor(env, n)?.clone());
            }
            env.insert(out1(node)?, Val::L(l));
            Ok(())
        }
        "append" => {
            let mut l = list(env, &i[0])?.clone();
            l.push(tensor(env, &i[1])?.clone());
            env.insert(out1(node)?, Val::L(l));
            Ok(())
        }
        "depth_attend" => {
            let mut src: Vec<Tensor> = list(env, &i[0])?.clone();
            for n in &i[1..] {
                src.push(tensor(env, n)?.clone());
            }
            if src.len() == 1 {
                let s0 = src.pop().unwrap();
                return put(env, node, s0);
            }
            let w = model.wa(node, "weight", app)?;
            // IR_SPEC 5.18 (1.2): eps is optional, 1e-5 by default
            let eps = if node.get("eps").is_some() { at_f32(node, "eps", a)? } else { 1e-5f32 };
            let (t_n, d) = (src[0].rows(), src[0].cols());
            if src.iter().any(|s| s.rows() != t_n || s.cols() != d) || w.numel() != d {
                return Err("depth_attend: sources or pseudo-query of different shapes".into());
            }
            let mut y = Vec::with_capacity(t_n * d);
            for t in 0..t_n {
                let mut l: Vec<f32> = src
                    .iter()
                    .map(|s| {
                        let r = s.row(t);
                        let mut ss = 0.0f32;
                        for &v in r {
                            ss += v * v;
                        }
                        let ri = 1.0f32 / (ss / d as f32 + eps).sqrt();
                        let mut acc = 0.0f32;
                        for j in 0..d {
                            acc += (r[j] * ri) * w.data[j];
                        }
                        acc
                    })
                    .collect();
                softmax_inplace(&mut l);
                for j in 0..d {
                    // s_0 a_0 + s_1 a_1 + ..., as written: the first product, then each sum
                    let mut acc = src[0].row(t)[j] * l[0];
                    for (si, s) in src.iter().enumerate().skip(1) {
                        acc += s.row(t)[j] * l[si];
                    }
                    y.push(acc);
                }
            }
            put(env, node, Tensor::mat(t_n, d, y))
        }
        "expr" => {
            let srcs: Vec<&Tensor> = i.iter().map(|n| tensor(env, n)).collect::<Result<_, _>>()?;
            let (t_n, w) = (srcs[0].rows(), srcs[0].cols());
            if srcs.iter().any(|s| s.rows() != t_n || s.cols() != w) {
                return Err("expr: inputs of different shapes".into());
            }
            let e = expr::parse_expr(at_str(node, "expr", a).ok_or("expr: no expr")?, srcs.len())?;
            let data: Vec<&[f32]> = srcs.iter().map(|s| s.data.as_slice()).collect();
            let y = expr::eval_expr(&e, &data, t_n, w);
            put(env, node, Tensor::mat(t_n, w, y))
        }
        "linear_attention" => {
            let q = input(0)?;
            let k = input(1)?;
            let v = input(2)?;
            let heads = at_usize(node, "heads", a)?;
            let hd = at_usize(node, "head_dim", a)?;
            let floor = at_f32(node, "den_floor", a)?;
            let c_n = heads * hd;
            if q.cols() != c_n || k.cols() != c_n || v.cols() != c_n {
                return Err("linear_attention: q, k, v are not [T, heads * head_dim]".into());
            }
            let st = rec_state(ctx, node, slots, &[vec![heads, hd], vec![heads, hd]])?;
            let mut y = Vec::with_capacity(q.numel());
            for t in 0..q.rows() {
                for c in 0..c_n {
                    let pk = phi(k.row(t)[c]);
                    st[0].data[c] += pk * v.row(t)[c];
                    st[1].data[c] += pk;
                    let den = expr::fmax(st[1].data[c], floor);
                    y.push((phi(q.row(t)[c]) * st[0].data[c]) / den);
                }
            }
            put(env, node, Tensor::mat(q.rows(), c_n, y))
        }
        "ssm_scan" => {
            let u = input(0)?;
            let n = at_usize(node, "n", a)?;
            let am = model.wa(node, "a", app)?.data.clone();
            let b = model.wa(node, "b", app)?.data.clone();
            let c = model.wa(node, "c", app)?.data.clone();
            let dd = model.wa(node, "d", app)?.data.clone();
            if u.cols() != n || am.len() != n * n || b.len() != n || c.len() != n || dd.len() != 1 {
                return Err("ssm_scan: tensors do not fit n".into());
            }
            let st = rec_state(ctx, node, slots, &[vec![n]])?;
            let mut y = Vec::with_capacity(u.numel());
            for t in 0..u.rows() {
                let ur = u.row(t);
                let prev = st[0].data.clone();
                for r in 0..n {
                    let mut s = 0.0f32;
                    for j in 0..n {
                        s += am[r * n + j] * prev[j];
                    }
                    st[0].data[r] = s + b[r] * ur[r];
                }
                for r in 0..n {
                    y.push(c[r] * st[0].data[r] + dd[0] * ur[r]);
                }
            }
            put(env, node, Tensor::mat(u.rows(), n, y))
        }
        "conv1d_causal" => {
            let x = input(0)?;
            let ch = at_usize(node, "channels", a)?;
            let kk = at_usize(node, "kernel", a)?;
            let w = model.wa(node, "weight", app)?.data.clone();
            let b = model.wa_opt(node, "bias", app)?.map(|t| t.data.clone());
            if x.cols() != ch || w.len() != ch * kk || b.as_ref().map(|b| b.len() != ch).unwrap_or(false) || kk < 2 {
                return Err("conv1d_causal: tensors do not fit channels and kernel".into());
            }
            let st = rec_state(ctx, node, slots, &[vec![kk - 1, ch]])?;
            let mut y = Vec::with_capacity(x.numel());
            for t in 0..x.rows() {
                // xx = the state's K - 1 rows, then this token's
                let mut xx = st[0].data.clone();
                xx.extend_from_slice(x.row(t));
                for c in 0..ch {
                    let mut s = 0.0f32;
                    for k in 0..kk {
                        s += w[c * kk + k] * xx[k * ch + c];
                    }
                    if let Some(b) = &b {
                        s += b[c];
                    }
                    y.push(s);
                }
                st[0].data = xx[ch..].to_vec();
            }
            put(env, node, Tensor::mat(x.rows(), ch, y))
        }
        "selective_scan" => {
            let x = input(0)?;
            let dt = input(1)?;
            let bm = input(2)?;
            let cm = input(3)?;
            let dch = at_usize(node, "channels", a)?;
            let ns = at_usize(node, "d_state", a)?;
            let soft = match at(node, "dt_softplus", a) {
                Some(v) => v.as_bool().unwrap_or(true),
                None => true,
            };
            let am = model.wa(node, "a", app)?.data.clone();
            let dk = model.wa_opt(node, "d", app)?.map(|t| t.data.clone());
            if x.cols() != dch || dt.cols() != dch || bm.cols() != ns || cm.cols() != ns || am.len() != dch * ns {
                return Err("selective_scan: inputs do not fit channels and d_state".into());
            }
            let st = rec_state(ctx, node, slots, &[vec![dch, ns]])?;
            let mut y = Vec::with_capacity(x.numel());
            for t in 0..x.rows() {
                for c in 0..dch {
                    let dv = dt.row(t)[c];
                    let delta = if !soft {
                        dv
                    } else if dv <= 20.0 {
                        dv.exp().ln_1p()
                    } else {
                        dv
                    };
                    let xd = x.row(t)[c] * delta;
                    let mut acc = 0.0f32;
                    for n in 0..ns {
                        let idx = c * ns + n;
                        let s = st[0].data[idx] * (delta * am[idx]).exp() + bm.row(t)[n] * xd;
                        st[0].data[idx] = s;
                        acc += s * cm.row(t)[n];
                    }
                    if let Some(dk) = &dk {
                        acc += dk[c] * x.row(t)[c];
                    }
                    y.push(acc);
                }
            }
            put(env, node, Tensor::mat(x.rows(), dch, y))
        }
        "recurrence" => {
            let rec = model.recs.get(&(ctx.section, ctx.node_index)).ok_or("recurrence: node not compiled")?;
            let srcs: Vec<&Tensor> = i.iter().map(|n| tensor(env, n)).collect::<Result<_, _>>()?;
            let t_n = srcs[0].rows();
            for (k, s) in srcs.iter().enumerate() {
                if s.cols() != rec.input_width(k) || s.rows() != t_n {
                    return Err(format!(
                        "IR_SPEC 6.10 (rule 34): input {} has width {}, its declaration {}",
                        i[k],
                        s.cols(),
                        rec.input_width(k)
                    ));
                }
            }
            let mut wts = Vec::new();
            for (tid, tn) in &rec.weights {
                let w = model.w(&model.tname(node, tn, app))?;
                let need: usize = rec.tensors[*tid].dims.iter().product();
                if w.numel() != need {
                    return Err(format!("IR_SPEC 6.10 (rule 34): weight {} holds {} elements, its declaration {}", tn, w.numel(), need));
                }
                wts.push(w.data.clone());
            }
            let slot = node.get("state").and_then(|s| s.as_str()).ok_or("recurrence: no state")?.to_string();
            let key = (slot.clone(), a);
            if !slots.rec.contains_key(&key) {
                let init = rec.initial_state(&wts);
                let ts = rec.state.iter().zip(init).map(|(&tid, v)| Tensor::new(rec.tensors[tid].dims.clone(), v)).collect();
                slots.rec.insert(key.clone(), ts);
            }
            let st = slots.rec.get_mut(&key).unwrap();
            let mut sv: Vec<Vec<f32>> = st.iter().map(|t| t.data.clone()).collect();
            let data: Vec<&[f32]> = srcs.iter().map(|s| s.data.as_slice()).collect();
            let y = rec.run(&data, t_n, &wts, &mut sv);
            for (t, v) in st.iter_mut().zip(sv) {
                t.data = v;
            }
            put(env, node, Tensor::mat(t_n, rec.out_width(), y))
        }
        "depth_route" => Err("depth_route outside the block runner".into()),
        other => Err(format!("op {} is not an op of this specification", other)),
    }
}

/// The recurrent slot instance a node reads, created at its initial value (+0).
fn rec_state<'a>(ctx: &Ctx, node: &Json, slots: &'a mut Slots, shapes: &[Vec<usize>]) -> Result<&'a mut Vec<Tensor>, String> {
    let slot = node.get("state").and_then(|s| s.as_str()).ok_or(format!("{}: no state", op(node)))?.to_string();
    let key = (slot, ctx.app.unwrap_or(0));
    let e = slots.rec.entry(key).or_insert_with(|| shapes.iter().map(|s| Tensor::zeros(s.clone())).collect());
    if e.len() != shapes.len() || e.iter().zip(shapes).any(|(t, s)| t.numel() != s.iter().product::<usize>()) {
        return Err(format!("{}: its slot's tensors do not fit the op", op(node)));
    }
    Ok(e)
}

fn attention(ctx: &Ctx, node: &Json, env: &mut Env, slots: &mut Slots) -> Result<(), String> {
    let a = ctx.app.unwrap_or(0);
    let i = ins(node);
    let q = tensor(env, &i[0])?;
    let k = tensor(env, &i[1])?;
    let v = tensor(env, &i[2])?;
    let heads = at_usize(node, "heads", a)?;
    let kvh = at_usize(node, "kv_heads", a)?;
    let hd = at_usize(node, "head_dim", a)?;
    let scale = at_f32(node, "scale", a)?;
    let window = at_i64(node, "window", a)?;
    let sinks = at_i64(node, "sinks", a)?;
    let turn = match at_str(node, "mask", a) {
        None | Some("causal") => false,
        Some("turn_state") => true,
        Some(m) => return Err(format!("attention: mask {}", m)),
    };
    if kvh == 0 || heads % kvh != 0 {
        return Err("attention: heads is not a multiple of kv_heads".into());
    }
    let t_n = q.rows();
    if q.cols() != heads * hd || k.cols() != kvh * hd || v.cols() != kvh * hd || k.rows() != t_n || v.rows() != t_n {
        return Err(format!("attention: q {:?}, k {:?}, v {:?} do not fit the heads", q.shape, k.shape, v.shape));
    }
    if ctx.rows.pos.len() != t_n {
        return Err("attention: rows and positions differ".into());
    }
    let pbias = position_bias(ctx, node, heads)?;
    let rkv = relative_kv(ctx, node, heads, hd)?;
    if pbias.is_some() && rkv.is_some() {
        return Err("attention: relative_kv and position_bias on one node (IR_SPEC 5.15, rule 40)".into());
    }
    let slot = node.get("state").and_then(|s| s.as_str()).ok_or("attention: no state")?.to_string();
    let max_len = ctx.model.slot(&slot).and_then(|s| at(s, "max_len", a)).and_then(|v| v.as_i64());
    let inst = slots.kv.entry((slot, a)).or_default();
    for t in 0..t_n {
        let p = ctx.rows.pos[t];
        if let Some(m) = max_len {
            if p as i64 >= m {
                return Err(format!("attention: position {} >= the slot's max_len {} (IR_SPEC 6.2)", p, m));
            }
        }
        // rows stay in ascending position
        let at_ix = inst.pos.partition_point(|&x| x < p);
        if at_ix < inst.pos.len() && inst.pos[at_ix] == p {
            return Err(format!("attention: the slot already holds position {}", p));
        }
        inst.pos.insert(at_ix, p);
        inst.k.insert(at_ix, k.row(t).to_vec());
        inst.v.insert(at_ix, v.row(t).to_vec());
    }
    let group = heads / kvh;
    let codes = ctx.codes;
    if turn && codes.is_none() {
        return Err("attention: mask turn_state without state-token codes".into());
    }
    let mut y = vec![0.0f32; t_n * heads * hd];
    let mut scores: Vec<f32> = Vec::new();
    let mut vis: Vec<usize> = Vec::new();
    for t in 0..t_n {
        let p = ctx.rows.pos[t];
        vis.clear();
        for (r, &j) in inst.pos.iter().enumerate() {
            if j > p {
                break;
            }
            if window > 0 {
                let ji = j as i64;
                if !(ji < sinks || ji >= p as i64 - window + 1) {
                    continue;
                }
            }
            if turn {
                let c = codes.unwrap();
                if !(c.s[j] == c.s[p] || c.c[j]) {
                    continue;
                }
            }
            vis.push(r);
        }
        for h in 0..heads {
            let g = h / group;
            let qh = &q.row(t)[h * hd..(h + 1) * hd];
            scores.clear();
            let mut m = f32::NEG_INFINITY;
            for &r in &vis {
                // p - j: positions, never rows (IR_SPEC 5.15, also under depth_route)
                let dist = p - inst.pos[r];
                let qk = dot(qh, &inst.k[r][g * hd..(g + 1) * hd]);
                // Shaw (1.4): ((q . k) + (q . rk[n])) * scale, each step rounded
                let s = match &rkv {
                    Some(rel) => (qk + dot(qh, rel.key_row(dist, h))) * scale,
                    None => qk * scale,
                };
                // T5 (1.3): s rounded, then s + W[b(p - j), h], rounded
                let s = match &pbias {
                    Some(pb) => s + pb.bias(dist, h),
                    None => s,
                };
                if s > m {
                    m = s;
                }
                scores.push(s);
            }
            let mut z = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                z += *s;
            }
            let out = &mut y[t * heads * hd + h * hd..t * heads * hd + (h + 1) * hd];
            for (n, &r) in vis.iter().enumerate() {
                let wgt = scores[n] / z;
                let vr = &inst.v[r][g * hd..(g + 1) * hd];
                for e in 0..hd {
                    out[e] += wgt * vr[e];
                }
            }
            // Shaw (1.4): out = (sum of a v_j) + (sum of a rv[n]), the second sum from +0 in ascending j
            if let Some(rv) = rkv.as_ref().filter(|rel| rel.value.is_some()) {
                let mut acc = vec![0.0f32; hd];
                for (n, &r) in vis.iter().enumerate() {
                    let wgt = scores[n] / z;
                    let rvr = rv.value_row(p - inst.pos[r], h);
                    for e in 0..hd {
                        acc[e] += wgt * rvr[e];
                    }
                }
                for e in 0..hd {
                    out[e] += acc[e];
                }
            }
        }
    }
    put(env, node, Tensor::mat(t_n, heads * hd, y))
}

/// T5's relative position bias (IR_SPEC 5.15, revision 1.3): W [B, heads] (ne {heads, B}) and the buckets.
struct PosBias<'a> {
    w: &'a Tensor,
    heads: usize,
    buckets: Vec<usize>,
}

impl PosBias<'_> {
    /// W[b(n), h], b(n) = buckets[min(n, D - 1)]
    fn bias(&self, n: usize, h: usize) -> f32 {
        let b = self.buckets[n.min(self.buckets.len() - 1)];
        self.w.data[b * self.heads + h]
    }
}

fn position_bias<'a>(ctx: &Ctx<'a>, node: &Json, heads: usize) -> Result<Option<PosBias<'a>>, String> {
    let pb = match node.get("position_bias") {
        None => return Ok(None),
        Some(x) => x,
    };
    let wn = pb.get("weight").and_then(|x| x.as_str()).ok_or("attention: position_bias has no weight (rule 39)")?;
    let w = ctx.model.w(&ctx.model.tname(node, wn, ctx.app))?;
    let buckets: Vec<usize> = match pb.get("buckets").and_then(|b| b.as_arr()) {
        Some(a) if !a.is_empty() => a
            .iter()
            .map(|x| x.as_i64().filter(|v| *v >= 0).map(|v| v as usize))
            .collect::<Option<Vec<usize>>>()
            .ok_or("attention: position_bias.buckets holds a value that is not an integer >= 0 (rule 39)")?,
        _ => return Err("attention: position_bias.buckets is not a non-empty list (rule 39)".into()),
    };
    let rows = if w.shape.len() == 2 { w.shape[0] } else { 0 };
    if w.shape.len() != 2 || w.shape[1] != heads || buckets.iter().any(|&b| b >= rows) {
        return Err(format!("attention: position_bias table {:?} is not [B, heads = {}] with B above every bucket (rule 39)", w.shape, heads));
    }
    Ok(Some(PosBias { w, heads, buckets }))
}

/// Shaw's relative key and value vectors (IR_SPEC 5.15, revision 1.4): tables [K + 1, w], w = head_dim
/// (shared by the heads) or heads * head_dim (head h at columns h * head_dim ...).
struct RelKv<'a> {
    key: &'a Tensor,
    value: Option<&'a Tensor>,
    k: usize,
    hd: usize,
}

impl RelKv<'_> {
    fn row<'b>(t: &'b Tensor, n: usize, h: usize, k: usize, hd: usize) -> &'b [f32] {
        let w = t.cols();
        let off = if w == hd { 0 } else { h * hd };
        let r = n.min(k) * w + off;
        &t.data[r..r + hd]
    }
    fn key_row(&self, dist: usize, h: usize) -> &[f32] {
        RelKv::row(self.key, dist, h, self.k, self.hd)
    }
    fn value_row(&self, dist: usize, h: usize) -> &[f32] {
        RelKv::row(self.value.expect("value table"), dist, h, self.k, self.hd)
    }
}

fn relative_kv<'a>(ctx: &Ctx<'a>, node: &Json, heads: usize, hd: usize) -> Result<Option<RelKv<'a>>, String> {
    let rel = match node.get("relative_kv") {
        None => return Ok(None),
        Some(x) => x,
    };
    let k = rel
        .get("max_distance")
        .and_then(|x| x.as_i64())
        .filter(|v| *v >= 0)
        .ok_or("attention: relative_kv.max_distance is not an integer >= 0 (rule 40)")? as usize;
    let table = |key: &str| -> Result<Option<&'a Tensor>, String> {
        match rel.get(key) {
            None | Some(Json::Null) => Ok(None),
            Some(Json::Str(n)) => {
                let t = ctx.model.w(&ctx.model.tname(node, n, ctx.app))?;
                let ok = t.shape.len() == 2 && t.shape[0] == k + 1 && (t.shape[1] == hd || t.shape[1] == heads * hd);
                if !ok {
                    return Err(format!(
                        "attention: relative_kv.{} {:?} is not [max_distance + 1 = {}, head_dim {} or heads * head_dim {}] (rule 40)",
                        key, t.shape, k + 1, hd, heads * hd
                    ));
                }
                Ok(Some(t))
            }
            Some(_) => Err(format!("attention: relative_kv.{} is neither a name nor null (rule 40)", key)),
        }
    };
    let key = table("key")?.ok_or("attention: relative_kv has no key (rule 40)")?;
    let value = table("value")?;
    Ok(Some(RelKv { key, value, k, hd }))
}
