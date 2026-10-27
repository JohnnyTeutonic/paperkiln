//! IR_SPEC Appendix D (v1.4.2): the third-party graphs a GGUF's metadata and tensor names build (D.1 to
//! D.6), and D.7's normal form, which rule 38 compares (section 3.1). Written from the appendix's text.

use crate::gguf::{Gguf, Val};
use crate::json::{json_eq, Json, Number};
use std::collections::HashMap;

/// D.1: the eight architectures that build a graph (and, with "paperkiln", rule 38's labels).
pub const ARCHS: [&str; 8] = ["llama", "mistral", "qwen2", "qwen3", "gpt2", "gptneox", "gemma3", "mamba"];

struct Meta<'a> {
    g: &'a Gguf,
    arch: &'a str,
}

/// A count or size: any GGUF integer type (D.1's types). A float, string, bool or array there is "a key of
/// the wrong type".
fn as_int(v: &Val) -> Option<i64> {
    Some(match *v {
        Val::U8(x) => x as i64,
        Val::I8(x) => x as i64,
        Val::U16(x) => x as i64,
        Val::I16(x) => x as i64,
        Val::U32(x) => x as i64,
        Val::I32(x) => x as i64,
        Val::U64(x) => x as i64,
        Val::I64(x) => x,
        _ => return None,
    })
}

/// A real key (an epsilon, a RoPE base, a scaling factor): a GGUF float, or an integer read exactly.
fn as_real(v: &Val) -> Option<f64> {
    match *v {
        Val::F32(x) => Some(x as f64),
        Val::F64(x) => Some(x),
        _ => as_int(v).map(|x| x as f64),
    }
}

impl Meta<'_> {
    fn raw(&self, key: &str) -> Option<&Val> {
        self.g.get(&format!("{}.{}", self.arch, key))
    }
    fn name(&self, key: &str) -> String {
        format!("{}.{}", self.arch, key)
    }
    fn int(&self, key: &str) -> Result<i64, String> {
        match self.raw(key) {
            None => Err(format!("{} is missing", self.name(key))),
            Some(v) => as_int(v).ok_or(format!("{} is not an integer", self.name(key))),
        }
    }
    fn int_or(&self, key: &str, d: i64) -> Result<i64, String> {
        match self.raw(key) {
            None => Ok(d),
            Some(v) => as_int(v).ok_or(format!("{} is not an integer", self.name(key))),
        }
    }
    fn real(&self, key: &str) -> Result<f64, String> {
        match self.raw(key) {
            None => Err(format!("{} is missing", self.name(key))),
            Some(v) => as_real(v).ok_or(format!("{} is not a number", self.name(key))),
        }
    }
    fn real_or(&self, key: &str, d: f64) -> Result<f64, String> {
        match self.raw(key) {
            None => Ok(d),
            Some(v) => as_real(v).ok_or(format!("{} is not a number", self.name(key))),
        }
    }
    /// D.1: a truth value is true when the key holds true or a non-zero number.
    fn truth_or(&self, key: &str, d: bool) -> Result<bool, String> {
        match self.raw(key) {
            None => Ok(d),
            Some(Val::Bool(b)) => Ok(*b),
            Some(v) => as_real(v).map(|x| x != 0.0).ok_or(format!("{} is not a truth value", self.name(key))),
        }
    }
    fn str_opt(&self, key: &str) -> Result<Option<String>, String> {
        match self.raw(key) {
            None => Ok(None),
            Some(v) => v.as_str().map(Some).ok_or(format!("{} is not a string", self.name(key))),
        }
    }
    fn has(&self, t: &str) -> bool {
        self.g.info(t).is_some()
    }
    /// D.1 "No RoPE scaling".
    fn no_rope_scaling(&self) -> Result<(), String> {
        match self.str_opt("rope.scaling.type")?.as_deref() {
            None | Some("none") => {}
            Some(t) => return Err(format!("RoPE scaling {:?}", t)),
        }
        if self.has("rope_freqs.weight") {
            return Err("the file has rope_freqs.weight (RoPE scaling)".into());
        }
        Ok(())
    }
}

fn s(x: &str) -> Json {
    Json::str(x)
}
fn n(x: f64) -> Json {
    Json::Num(Number { v: x, int_form: x.fract() == 0.0 })
}
fn i(x: i64) -> Json {
    Json::int(x)
}
fn per_layer(v: Vec<Json>) -> Json {
    Json::obj(vec![("per_layer", Json::Arr(v))])
}

/// `op(in, ...) -> out {attributes}` (D.2).
fn node(op: &str, ins: &[&str], out: &[&str], attrs: Vec<(&str, Json)>) -> Json {
    let mut m: Vec<(String, Json)> = vec![("op".into(), s(op)), ("in".into(), Json::Arr(ins.iter().map(|x| s(x)).collect()))];
    for (k, v) in attrs {
        m.push((k.to_string(), v));
    }
    let o = if out.len() == 1 { s(out[0]) } else { Json::Arr(out.iter().map(|x| s(x)).collect()) };
    m.push(("out".into(), o));
    Json::Obj(m)
}

fn b(t: &str) -> String {
    format!("blk.{{i}}.{}", t)
}

fn rms(x: &str, w: &str, out: &str, eps: f64) -> Json {
    node("rmsnorm", &[x], &[out], vec![("weight", s(w)), ("eps", n(eps))])
}
fn rms_g(x: &str, w: &str, out: &str, eps: f64, hd: i64) -> Json {
    node("rmsnorm", &[x], &[out], vec![("weight", s(w)), ("eps", n(eps)), ("group", i(hd))])
}
fn ln(x: &str, p: &str, out: &str, eps: f64) -> Json {
    node("layernorm", &[x], &[out], vec![("weight", s(&format!("{}.weight", p))), ("bias", s(&format!("{}.bias", p))), ("eps", n(eps))])
}
fn lin(x: &str, w: &str, bias: Option<&str>, out: &str) -> Json {
    node("linear", &[x], &[out], vec![("weight", s(w)), ("bias", bias.map(s).unwrap_or(Json::Null))])
}

struct Common {
    d: i64,
    l: i64,
    h: i64,
    kv: i64,
    hd: i64,
    ctx: i64,
    v: i64,
}

fn rope(c: &Common, theta: Json, pairing: &str, n_rot: i64, freq_scale: Option<Json>) -> Json {
    let mut a = vec![
        ("heads", i(c.h)),
        ("kv_heads", i(c.kv)),
        ("head_dim", i(c.hd)),
        ("n_rot", i(n_rot)),
        ("theta", theta),
        ("rope_heads", s("all")),
        ("pairing", s(pairing)),
    ];
    if let Some(f) = freq_scale {
        a.push(("freq_scale", f));
    }
    node("rope", &["q", "k", "pos"], &["q", "k"], a)
}
fn att(c: &Common, scale: f64, w: Json) -> Json {
    node(
        "attention",
        &["q", "k", "v"],
        &["a"],
        vec![
            ("state", s("kv")),
            ("heads", i(c.h)),
            ("kv_heads", i(c.kv)),
            ("head_dim", i(c.hd)),
            ("causal", Json::Bool(true)),
            ("scale", n(scale)),
            ("window", w),
            ("sinks", i(0)),
        ],
    )
}
fn kv_slot(c: &Common, w: Json) -> Json {
    Json::obj(vec![
        ("name", s("kv")),
        ("kind", s("kv")),
        ("heads", i(c.h)),
        ("kv_heads", i(c.kv)),
        ("head_dim", i(c.hd)),
        ("max_len", i(c.ctx)),
        ("window", w),
        ("sinks", i(0)),
        ("dtype", s("f32")),
    ])
}
fn e0() -> Json {
    node("embedding", &["tokens"], &["x"], vec![("weight", s("token_embd.weight"))])
}
fn head_lin(m: &Meta) -> Json {
    let mut h = lin("h", "output.weight", None, "logits");
    if !m.has("output.weight") {
        if let Json::Obj(v) = &mut h {
            let at = v.len() - 1;
            v.insert(at, ("tied_to".into(), s("token_embd.weight")));
        }
    }
    h
}

/// D.1: V, the vocabulary size.
fn vocab(g: &Gguf) -> Result<i64, String> {
    if let Some(Val::Arr(a)) = g.get("tokenizer.ggml.tokens") {
        if !a.is_empty() {
            return Ok(a.len() as i64);
        }
    }
    match g.info("token_embd.weight") {
        Some(t) if t.ne.len() >= 2 => Ok(t.ne[1] as i64),
        _ => Err("neither tokenizer.ggml.tokens nor a token_embd.weight with ne[1]".into()),
    }
}

fn common(m: &Meta, v: i64) -> Result<Common, String> {
    let d = m.int("embedding_length")?;
    let l = m.int("block_count")?;
    let h = m.int("attention.head_count")?;
    if h == 0 {
        return Err("a head count of 0".into());
    }
    // D.1 (v1.4.2): only H = 0 builds no graph; any other zero or negative count builds one
    let kv = m.int_or("attention.head_count_kv", h)?;
    let hd = m.int_or("attention.key_length", d / h)?;
    let ctx = m.int("context_length")?;
    Ok(Common { d, l, h, kv, hd, ctx, v })
}

fn doc(arch: &str, c: &Common, e: Vec<Json>, st: Vec<Json>, nodes: Vec<Json>, hd: Vec<Json>) -> Json {
    Json::obj(vec![
        ("ir_version", i(1)),
        ("family_hint", s(arch)),
        ("model", Json::obj(vec![("vocab", i(c.v)), ("d", i(c.d)), ("n_layers", i(c.l)), ("ctx", i(c.ctx)), ("dtype", s("f32"))])),
        ("embed", Json::obj(vec![("nodes", Json::Arr(e))])),
        ("block", Json::obj(vec![("repeat", i(c.l)), ("state", Json::Arr(st)), ("nodes", Json::Arr(nodes))])),
        ("head", Json::obj(vec![("nodes", Json::Arr(hd))])),
    ])
}

/// The document Appendix D builds from the file, or why the metadata builds no graph.
pub fn build(g: &Gguf) -> Result<Json, String> {
    let arch = g.get_str("general.architecture").ok_or("no general.architecture")?;
    if !ARCHS.contains(&arch.as_str()) {
        return Err(format!("{:?} is not an architecture of Appendix D", arch));
    }
    let m = Meta { g, arch: &arch };
    let v = vocab(g)?;
    match arch.as_str() {
        "llama" | "mistral" | "qwen2" | "qwen3" => d3(&m, v),
        "gpt2" | "gptneox" => d4(&m, v),
        "gemma3" => d5(&m, v),
        _ => d6(&m, v),
    }
}

/// D.3 llama, mistral, qwen2, qwen3.
fn d3(m: &Meta, v: i64) -> Result<Json, String> {
    let c = common(m, v)?;
    let eps = m.real("attention.layer_norm_rms_epsilon")?;
    let theta = m.real_or("rope.freq_base", 10000.0)?;
    let n_rot = m.int_or("rope.dimension_count", c.hd)?;
    m.no_rope_scaling()?;
    let qwen = m.arch == "qwen2" || m.arch == "qwen3";
    let pairing = if qwen { "neox" } else { "interleaved" };
    let w = if qwen { 0 } else { m.int_or("attention.sliding_window", 0)? };
    let bias = m.has("blk.0.attn_q.bias");
    let qkn = m.has("blk.0.attn_q_norm.weight");
    let (qb, kb, vb) = (b("attn_q.bias"), b("attn_k.bias"), b("attn_v.bias"));
    let mut nd = vec![
        rms("x", &b("attn_norm.weight"), "h", eps),
        lin("h", &b("attn_q.weight"), if bias { Some(&qb) } else { None }, "q"),
        lin("h", &b("attn_k.weight"), if bias { Some(&kb) } else { None }, "k"),
        lin("h", &b("attn_v.weight"), if bias { Some(&vb) } else { None }, "v"),
    ];
    if qkn {
        nd.push(rms_g("q", &b("attn_q_norm.weight"), "q", eps, c.hd));
        nd.push(rms_g("k", &b("attn_k_norm.weight"), "k", eps, c.hd));
    }
    nd.push(rope(&c, n(theta), pairing, n_rot, None));
    nd.push(att(&c, 1.0 / (c.hd as f64).sqrt(), i(w)));
    nd.push(lin("a", &b("attn_output.weight"), None, "o"));
    nd.push(node("add", &["x", "o"], &["x"], vec![]));
    nd.push(rms("x", &b("ffn_norm.weight"), "h2", eps));
    let ec = if m.arch == "llama" { m.int_or("expert_count", 0)? } else { 0 };
    if m.arch == "llama" && ec != 0 && m.has("blk.0.ffn_gate_inp.weight") {
        nd.push(node(
            "moe_ffn",
            &["h2"],
            &["f"],
            vec![
                ("router", s(&b("ffn_gate_inp.weight"))),
                (
                    "experts",
                    Json::obj(vec![("gate", s(&b("ffn_gate_exps.weight"))), ("up", s(&b("ffn_up_exps.weight"))), ("down", s(&b("ffn_down_exps.weight")))]),
                ),
                ("n_experts", i(ec)),
                ("top_k", i(m.int("expert_used_count")?)),
                ("gate_norm", s("renorm")),
                ("activation", s("silu")),
            ],
        ));
    } else {
        nd.push(lin("h2", &b("ffn_gate.weight"), None, "g"));
        nd.push(lin("h2", &b("ffn_up.weight"), None, "u"));
        nd.push(node("silu", &["g"], &["g"], vec![]));
        nd.push(node("mul", &["g", "u"], &["gu"], vec![]));
        nd.push(lin("gu", &b("ffn_down.weight"), None, "f"));
    }
    nd.push(node("add", &["x", "f"], &["x"], vec![]));
    let hd = vec![rms("x", "output_norm.weight", "h", eps), head_lin(m)];
    Ok(doc(m.arch, &c, vec![e0()], vec![kv_slot(&c, i(w))], nd, hd))
}

/// D.4 gpt2, gptneox.
fn d4(m: &Meta, v: i64) -> Result<Json, String> {
    let c = common(m, v)?;
    let eps = m.real("attention.layer_norm_epsilon")?;
    let neox = m.arch == "gptneox";
    let (theta, n_rot, par) = if neox {
        m.no_rope_scaling()?;
        (m.real_or("rope.freq_base", 10000.0)?, m.int_or("rope.dimension_count", c.hd)?, m.truth_or("use_parallel_residual", true)?)
    } else {
        (0.0, 0, false)
    };
    let mut e = vec![e0()];
    if !neox {
        e.push(node("pos_embedding", &["x", "pos"], &["x"], vec![("weight", s("position_embd.weight"))]));
    }
    let mut nd = vec![
        ln("x", &b("attn_norm"), "h", eps),
        lin("h", &b("attn_qkv.weight"), Some(&b("attn_qkv.bias")), "qkv"),
        node("split", &["qkv"], &["q", "k", "v"], vec![("sizes", Json::Arr(vec![i(c.h * c.hd), i(c.kv * c.hd), i(c.kv * c.hd)]))]),
    ];
    if neox {
        nd.push(rope(&c, n(theta), "neox", n_rot, None));
    }
    nd.push(att(&c, 1.0 / (c.hd as f64).sqrt(), i(0)));
    nd.push(lin("a", &b("attn_output.weight"), Some(&b("attn_output.bias")), "o"));
    let f = vec![
        ln("x", &b("ffn_norm"), "h2", eps),
        lin("h2", &b("ffn_up.weight"), Some(&b("ffn_up.bias")), "u"),
        node("gelu", &["u"], &["u"], vec![("approx", s(if neox { "none" } else { "tanh" }))]),
        lin("u", &b("ffn_down.weight"), Some(&b("ffn_down.bias")), "f"),
    ];
    if neox && par {
        nd.extend(f);
        nd.push(node("add", &["f", "x"], &["x"], vec![]));
        nd.push(node("add", &["x", "o"], &["x"], vec![]));
    } else {
        nd.push(node("add", &["x", "o"], &["x"], vec![]));
        nd.extend(f);
        nd.push(node("add", &["x", "f"], &["x"], vec![]));
    }
    let hd = if neox {
        vec![ln("x", "output_norm", "h", eps), lin("h", "output.weight", None, "logits")]
    } else {
        vec![ln("x", "output_norm", "h", eps), head_lin(m)]
    };
    Ok(doc(m.arch, &c, e, vec![kv_slot(&c, i(0))], nd, hd))
}

/// D.5 gemma3.
fn d5(m: &Meta, v: i64) -> Result<Json, String> {
    let c = common(m, v)?;
    let eps = m.real("attention.layer_norm_rms_epsilon")?;
    let n_rot = m.int_or("rope.dimension_count", c.hd)?;
    if m.has("rope_freqs.weight") {
        return Err("the file has rope_freqs.weight".into());
    }
    let ty = m.str_opt("rope.scaling.type")?;
    let cfac = match ty.as_deref() {
        None | Some("none") => 1.0,
        Some("linear") => {
            let f = m.real_or("rope.scaling.factor", 1.0)?;
            if f == 0.0 {
                1.0
            } else {
                f
            }
        }
        Some(t) => return Err(format!("RoPE scaling {:?}", t)),
    };
    let sw = m.int("attention.sliding_window")?;
    let l = c.l.max(0) as usize;
    let local = |a: usize| a % 6 < 5;
    let g = if (0..l).any(|a| !local(a)) { m.real("rope.freq_base")? } else { 0.0 };
    let windows = per_layer((0..l).map(|a| i(if local(a) { sw } else { 0 })).collect());
    let thetas = per_layer((0..l).map(|a| n(if local(a) { 10000.0 } else { g })).collect());
    let fs = if cfac != 1.0 { Some(per_layer((0..l).map(|a| n(if local(a) { 1.0 } else { 1.0 / cfac })).collect())) } else { None };
    let scale = if c.l == 62 { 1.0 / (c.d as f64 / c.h as f64).sqrt() } else { 1.0 / (c.hd as f64).sqrt() };
    let e = vec![e0(), node("affine", &["x"], &["x"], vec![("scale", n((c.d as f64).sqrt())), ("shift", i(0))])];
    let nd = vec![
        rms("x", &b("attn_norm.weight"), "h", eps),
        lin("h", &b("attn_q.weight"), None, "q"),
        lin("h", &b("attn_k.weight"), None, "k"),
        lin("h", &b("attn_v.weight"), None, "v"),
        rms_g("q", &b("attn_q_norm.weight"), "q", eps, c.hd),
        rms_g("k", &b("attn_k_norm.weight"), "k", eps, c.hd),
        rope(&c, thetas, "neox", n_rot, fs),
        att(&c, scale, windows.clone()),
        lin("a", &b("attn_output.weight"), None, "o"),
        rms("o", &b("post_attention_norm.weight"), "o", eps),
        node("add", &["x", "o"], &["x"], vec![]),
        rms("x", &b("ffn_norm.weight"), "h2", eps),
        lin("h2", &b("ffn_gate.weight"), None, "g"),
        lin("h2", &b("ffn_up.weight"), None, "u"),
        node("gelu", &["g"], &["g"], vec![("approx", s("tanh"))]),
        node("mul", &["g", "u"], &["gu"], vec![]),
        lin("gu", &b("ffn_down.weight"), None, "f"),
        rms("f", &b("post_ffw_norm.weight"), "f", eps),
        node("add", &["x", "f"], &["x"], vec![]),
    ];
    let hd = vec![rms("x", "output_norm.weight", "h", eps), head_lin(m)];
    Ok(doc(m.arch, &c, e, vec![kv_slot(&c, windows)], nd, hd))
}

/// D.6 mamba (Mamba-1).
fn d6(m: &Meta, v: i64) -> Result<Json, String> {
    let kc = m.int("ssm.conv_kernel")?;
    let di = m.int("ssm.inner_size")?;
    let ns = m.int("ssm.state_size")?;
    let r = m.int("ssm.time_step_rank")?;
    let eps = m.real("attention.layer_norm_rms_epsilon")?;
    let l = m.int("block_count")?;
    let d = m.int("embedding_length")?;
    let ctx = match m.int_or("context_length", 0)? {
        0 => 1048576,
        x => x,
    };
    if !m.has("blk.0.ssm_dt.weight") || !m.has("blk.0.ssm_d") {
        return Err("the file lacks blk.0.ssm_dt.weight or blk.0.ssm_d".into());
    }
    if m.has("blk.0.ssm_dt_norm.weight") || m.truth_or("ssm.dt_b_c_rms", false)? {
        return Err("Falcon-Mamba (ssm_dt_norm or ssm.dt_b_c_rms)".into());
    }
    let c = Common { d, l, h: 0, kv: 0, hd: 0, ctx, v };
    let rec = |name: &str, op: &str, t: &str, shape: Vec<i64>| {
        Json::obj(vec![
            ("name", s(name)),
            ("kind", s("recurrent")),
            ("op", s(op)),
            ("dtype", s("f32")),
            ("tensors", Json::Arr(vec![Json::obj(vec![("name", s(t)), ("shape", Json::Arr(shape.into_iter().map(i).collect())), ("init", i(0))])])),
        ])
    };
    let st = vec![rec("conv", "conv1d_causal", "x", vec![kc - 1, di]), rec("ssm", "selective_scan", "s", vec![di, ns])];
    let nd = vec![
        rms("x", &b("attn_norm.weight"), "h", eps),
        lin("h", &b("ssm_in.weight"), None, "xz"),
        node("split", &["xz"], &["xs", "z"], vec![("sizes", Json::Arr(vec![i(di), i(di)]))]),
        node(
            "conv1d_causal",
            &["xs"],
            &["xs"],
            vec![("state", s("conv")), ("weight", s(&b("ssm_conv1d.weight"))), ("bias", s(&b("ssm_conv1d.bias"))), ("channels", i(di)), ("kernel", i(kc))],
        ),
        node("silu", &["xs"], &["xs"], vec![]),
        lin("xs", &b("ssm_x.weight"), None, "xdb"),
        node("split", &["xdb"], &["dt", "bm", "cm"], vec![("sizes", Json::Arr(vec![i(r), i(ns), i(ns)]))]),
        lin("dt", &b("ssm_dt.weight"), Some(&b("ssm_dt.bias")), "dt"),
        node(
            "selective_scan",
            &["xs", "dt", "bm", "cm"],
            &["y"],
            vec![
                ("state", s("ssm")),
                ("a", s(&b("ssm_a"))),
                ("d", s(&b("ssm_d"))),
                ("channels", i(di)),
                ("d_state", i(ns)),
                ("dt_softplus", Json::Bool(true)),
            ],
        ),
        node("silu", &["z"], &["z"], vec![]),
        node("mul", &["y", "z"], &["y"], vec![]),
        lin("y", &b("ssm_out.weight"), None, "o"),
        node("add", &["x", "o"], &["x"], vec![]),
    ];
    let hd = vec![rms("x", "output_norm.weight", "h", eps), head_lin(m)];
    Ok(doc(m.arch, &c, vec![e0()], st, nd, hd))
}

// ---------------------------------------------------------------- D.7

/// `_` members removed at any depth and, of two equal member names, the last kept (3.2).
fn clean(v: &Json) -> Json {
    match v {
        Json::Obj(m) => {
            let mut out: Vec<(String, Json)> = Vec::new();
            for (k, x) in m {
                if k.starts_with('_') {
                    continue;
                }
                let x = clean(x);
                if let Some(p) = out.iter().position(|(ok, _)| ok == k) {
                    out.remove(p);
                }
                out.push((k.clone(), x));
            }
            Json::Obj(out)
        }
        Json::Arr(a) => Json::Arr(a.iter().map(clean).collect()),
        x => x.clone(),
    }
}

/// Step 6: every number is the f32 it rounds to.
fn f32_numbers(v: &Json) -> Json {
    match v {
        Json::Num(x) => Json::Num(Number { v: (x.v as f32) as f64, int_form: false }),
        Json::Obj(m) => Json::Obj(m.iter().map(|(k, x)| (k.clone(), f32_numbers(x))).collect()),
        Json::Arr(a) => Json::Arr(a.iter().map(f32_numbers).collect()),
        x => x.clone(),
    }
}

fn set(m: &mut Vec<(String, Json)>, k: &str, v: Json) {
    match m.iter_mut().find(|(ok, _)| ok == k) {
        Some(e) => e.1 = v,
        None => m.push((k.to_string(), v)),
    }
}
fn fill(m: &mut Vec<(String, Json)>, k: &str, v: Json) {
    if !m.iter().any(|(ok, _)| ok == k) {
        m.push((k.to_string(), v));
    }
}
fn member(m: &[(String, Json)], k: &str) -> Option<Json> {
    m.iter().find(|(ok, _)| ok == k).map(|(_, v)| v.clone())
}

struct Walk<'a> {
    writes: HashMap<String, String>,
    count: usize,
    slots: Vec<String>,
    has: &'a dyn Fn(&str) -> bool,
}

impl Walk<'_> {
    fn node(&mut self, n: &Json) -> Json {
        let mut m = match n {
            Json::Obj(m) => m.clone(),
            x => return x.clone(),
        };
        let op = member(&m, "op").and_then(|o| o.as_str().map(String::from)).unwrap_or_default();
        // step 1: the informative attributes
        match op.as_str() {
            "attention" => m.retain(|(k, _)| k != "softmax"),
            "moe_ffn" => m.retain(|(k, _)| k != "routing"),
            _ => {}
        }
        // step 2: optional attributes at their defaults (rmsnorm.group excepted)
        match op.as_str() {
            "linear" => {
                fill(&mut m, "bias", Json::Null);
                fill(&mut m, "tied_to", Json::Null);
            }
            "scale" => fill(&mut m, "bias", Json::Null),
            "rope" => {
                fill(&mut m, "freq_scale", i(1));
                if let Some(h) = member(&m, "heads") {
                    fill(&mut m, "kv_heads", h);
                }
                if let Some(h) = member(&m, "head_dim") {
                    fill(&mut m, "n_rot", h);
                }
            }
            "attention" => fill(&mut m, "mask", s("causal")),
            "rmsnorm" => fill(&mut m, "weight_offset", i(0)),
            "depth_attend" => fill(&mut m, "eps", self::n(1e-5)),
            "conv1d_causal" => fill(&mut m, "bias", Json::Null),
            "selective_scan" => {
                fill(&mut m, "d", Json::Null);
                fill(&mut m, "dt_softplus", Json::Bool(true));
            }
            _ => {}
        }
        // step 3: tied_to
        if op == "linear" {
            if let Some(Json::Str(t)) = member(&m, "tied_to") {
                let w = member(&m, "weight").and_then(|w| w.as_str().map(String::from)).unwrap_or_default();
                if !(self.has)(&w) {
                    set(&mut m, "weight", s(&t));
                }
                set(&mut m, "tied_to", Json::Null);
            }
        }
        // step 4: slots by index
        if let Some(Json::Str(sn)) = member(&m, "state") {
            let nm = match self.slots.iter().position(|x| *x == sn) {
                Some(k) => format!("slot{}", k),
                None => format!("undeclared:{}", sn),
            };
            set(&mut m, "state", s(&nm));
        }
        // step 5: values by their writes
        if let Some(Json::Arr(ins)) = member(&m, "in") {
            let mut ids: Vec<String> = ins
                .iter()
                .map(|x| {
                    let nm = x.as_str().unwrap_or("");
                    if nm == "tokens" || nm == "pos" {
                        nm.to_string()
                    } else {
                        self.writes.get(nm).cloned().unwrap_or(format!("unwritten:{}", nm))
                    }
                })
                .collect();
            if (op == "add" || op == "mul") && ids.len() == 2 {
                ids.sort();
            }
            set(&mut m, "in", Json::Arr(ids.iter().map(|x| s(x)).collect()));
        }
        let new_out = |nm: &str, w: &mut Self| -> Json {
            w.count += 1;
            let id = format!("v{}", w.count);
            w.writes.insert(nm.to_string(), id.clone());
            s(&id)
        };
        match member(&m, "out") {
            Some(Json::Str(o)) => {
                let j = new_out(&o, self);
                set(&mut m, "out", j);
            }
            Some(Json::Arr(os)) => {
                let js: Vec<Json> = os.iter().map(|o| new_out(o.as_str().unwrap_or(""), self)).collect();
                set(&mut m, "out", Json::Arr(js));
            }
            _ => {}
        }
        Json::Obj(m)
    }
}

/// D.7's normal form of a document, given the file's tensor names.
pub fn normal_form(doc: &Json, has: &dyn Fn(&str) -> bool) -> Json {
    let d = clean(doc);
    let block = d.get("block").cloned().unwrap_or(Json::Null);
    let slots: Vec<Json> = block.get("state").and_then(|x| x.as_arr()).cloned().unwrap_or_default();
    let slot_names: Vec<String> = slots.iter().map(|x| x.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string()).collect();
    let mut w = Walk { writes: HashMap::new(), count: 0, slots: slot_names, has };
    let mut out: Vec<(String, Json)> = Vec::new();
    if let Some(mo) = d.get("model") {
        out.push(("model".into(), mo.clone()));
    }
    for sec in ["embed", "block", "head"] {
        let so = match d.get(sec) {
            Some(Json::Obj(m)) => m.clone(),
            Some(x) => {
                out.push((sec.into(), x.clone()));
                continue;
            }
            None => continue,
        };
        let mut so = so;
        if let Some(Json::Arr(nodes)) = member(&so, "nodes") {
            let nn: Vec<Json> = nodes.iter().map(|x| w.node(x)).collect();
            set(&mut so, "nodes", Json::Arr(nn));
        }
        if sec == "block" {
            if let Some(Json::Arr(st)) = member(&so, "state") {
                let ns: Vec<Json> = st
                    .iter()
                    .enumerate()
                    .map(|(k, sl)| match sl {
                        Json::Obj(m) => {
                            let mut m = m.clone();
                            set(&mut m, "name", s(&format!("slot{}", k)));
                            if member(&m, "kind").and_then(|x| x.as_str().map(String::from)).as_deref() == Some("kv") {
                                fill(&mut m, "dtype", s("f32"));
                            }
                            Json::Obj(m)
                        }
                        x => x.clone(),
                    })
                    .collect();
                set(&mut so, "state", Json::Arr(ns));
            }
        }
        out.push((sec.into(), Json::Obj(so)));
    }
    f32_numbers(&Json::Obj(out))
}

/// The first place two normal forms differ, for a refusal's message.
pub fn first_diff(a: &Json, b: &Json, path: &str) -> Option<String> {
    if json_eq(a, b) {
        return None;
    }
    match (a, b) {
        (Json::Obj(x), Json::Obj(y)) => {
            for (k, v) in x {
                match y.iter().find(|(k2, _)| k2 == k) {
                    None => return Some(format!("{}.{}: only in the document", path, k)),
                    Some((_, v2)) => {
                        if let Some(d) = first_diff(v, v2, &format!("{}.{}", path, k)) {
                            return Some(d);
                        }
                    }
                }
            }
            for (k, _) in y {
                if !x.iter().any(|(k2, _)| k2 == k) {
                    return Some(format!("{}.{}: only in Appendix D's graph", path, k));
                }
            }
            None
        }
        (Json::Arr(x), Json::Arr(y)) => {
            for (j, (p, q)) in x.iter().zip(y).enumerate() {
                if let Some(d) = first_diff(p, q, &format!("{}[{}]", path, j)) {
                    return Some(d);
                }
            }
            Some(format!("{}: {} entries in the document, {} in Appendix D's graph", path, x.len(), y.len()))
        }
        _ => Some(format!("{}: {} in the document, {} in Appendix D's graph", path, short(a), short(b))),
    }
}

fn short(v: &Json) -> String {
    let t = crate::json::to_string(v);
    if t.chars().count() > 120 {
        format!("{}...", t.chars().take(120).collect::<String>())
    } else {
        t
    }
}
