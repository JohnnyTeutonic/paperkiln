//! Load-time refusals (IR_SPEC 10.1): the schema (rule 1) and every semantic and container rule.

use crate::digest;
use crate::gguf::Gguf;
use crate::json::Json;
use crate::model::{at, ins, is_per_layer, op, outs, STATEFUL};
use crate::recurrence::Rec;
use crate::schema::Schema;

const NODE_STRUCTURAL: [&str; 26] = [
    "op", "in", "out", "state", "tensor_index", "sizes", "expr", "position_bias", "relative_kv", "index", "inputs", "weights", "init",
    "let", "update", "readout", "weight", "bias", "router", "experts", "state_weight", "tied_to", "a", "b", "c", "d",
];
const SLOT_STRUCTURAL: [&str; 7] = ["name", "kind", "op", "dtype", "tensors", "evict", "kv_rows"];
const TENSOR_KEYS: [&str; 10] = ["weight", "bias", "router", "state_weight", "tied_to", "a", "b", "c", "d", "experts"];

/// Closed sets (rule 5), including inside per_layer lists.
const CLOSED: [(&str, &str, &[&str]); 7] = [
    ("gelu", "approx", &["tanh", "none"]),
    ("rope", "pairing", &["interleaved", "neox"]),
    ("rope", "rope_heads", &["all", "first"]),
    ("attention", "mask", &["causal", "turn_state"]),
    ("moe_ffn", "gate_norm", &["renorm", "raw"]),
    ("moe_ffn", "activation", &["silu"]),
    ("linear_attention", "feature_map", &["elu_plus_one"]),
];
const INT_ATTRS: [&str; 14] =
    ["heads", "kv_heads", "head_dim", "window", "sinks", "n_rot", "group", "n_experts", "top_k", "n", "channels", "kernel", "d_state", "max_len"];
const NUM_ATTRS: [&str; 7] = ["eps", "theta", "freq_scale", "scale", "shift", "den_floor", "weight_offset"];
const BOOL_ATTRS: [&str; 2] = ["causal", "dt_softplus"];

pub struct Report {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

struct V {
    errors: Vec<String>,
    warnings: Vec<String>,
}

impl V {
    fn e(&mut self, rule: &str, sec: &str, m: String) {
        self.errors.push(format!("IR_SPEC {} (rule {}): {}", sec, rule, m));
    }
}

fn tensor_names(node: &Json) -> Vec<(String, String)> {
    let mut v = Vec::new();
    for k in TENSOR_KEYS {
        match node.get(k) {
            Some(Json::Str(s)) => v.push((k.to_string(), s.clone())),
            Some(Json::Obj(m)) if k == "experts" => {
                for (ek, ev) in m {
                    if let Some(s) = ev.as_str() {
                        v.push((format!("experts.{}", ek), s.to_string()));
                    }
                }
            }
            _ => {}
        }
    }
    // IR_SPEC 3.7 (1.3, 1.4): position_bias's weight, relative_kv's key and value
    for (attr, keys) in [("position_bias", &["weight"][..]), ("relative_kv", &["key", "value"][..])] {
        if let Some(a) = node.get(attr) {
            for k in keys {
                if let Some(Json::Str(s)) = a.get(k) {
                    v.push((format!("{}.{}", attr, k), s.clone()));
                }
            }
        }
    }
    if op(node) == "recurrence" {
        if let Some(Json::Obj(m)) = node.get("weights") {
            for (dk, dv) in m {
                if let Some(s) = dv.as_str() {
                    v.push((format!("weights.{}", dk), s.to_string()));
                }
            }
        }
    }
    v
}

fn arity(o: &str) -> Option<(usize, usize, usize)> {
    // (min in, max in, outs)
    Some(match o {
        "sub" | "add" | "mul" | "mul_col" | "append" | "pos_embedding" => (2, 2, 1),
        "embedding" | "relu" | "sigmoid" | "silu" | "gelu" | "affine" | "rms_row" | "scale" | "rmsnorm" | "layernorm" | "linear"
        | "moe_ffn" | "ssm_scan" | "conv1d_causal" | "depth_route" => (1, 1, 1),
        "attention" | "linear_attention" => (3, 3, 1),
        "rope" => (3, 3, 2),
        "selective_scan" => (4, 4, 1),
        "list" | "depth_attend" | "recurrence" => (1, usize::MAX, 1),
        "expr" => (1, 23, 1),
        "split" => (1, 1, 0),
        _ => return None,
    })
}

pub fn validate(doc_raw: &Json, file: Option<&Gguf>) -> Report {
    let mut v = V { errors: Vec::new(), warnings: Vec::new() };
    let schema = Schema::load();
    for e in schema.validate(doc_raw) {
        // IR_SPEC 10.1 rule 1: a position bias and relative vectors are rules 39 and 40 (their own checks
        // below restate the schema's structure for them), so the schema's findings inside them are not rule 1
        let path = e.split(": ").next().unwrap_or("");
        if path.contains(".position_bias") || path.contains(".relative_kv") {
            continue;
        }
        v.e("1", "10.1", e);
    }
    if !matches!(doc_raw, Json::Obj(_)) {
        return Report { errors: v.errors, warnings: v.warnings };
    }
    let doc = crate::json::strip_prose(doc_raw);
    semantic(&doc, &mut v);
    if let Some(g) = file {
        container(&doc, g, &mut v);
    }
    Report { errors: v.errors, warnings: v.warnings }
}

fn nodes<'a>(doc: &'a Json, sec: &str) -> Vec<&'a Json> {
    doc.get(sec).and_then(|s| s.get("nodes")).and_then(|n| n.as_arr()).map(|a| a.iter().collect()).unwrap_or_default()
}

fn semantic(doc: &Json, v: &mut V) {
    // rule 2
    match doc.get("ir_version").and_then(|x| x.as_i64()) {
        Some(0) | Some(1) => {}
        other => v.e("2", "2.3", format!("ir_version {:?} is not a version this engine reads (it reads 0 and 1)", other)),
    }
    let model = doc.get("model");
    let posint = |k: &str| model.and_then(|m| m.get(k)).and_then(|x| x.as_i64()).filter(|x| *x >= 1);
    for k in ["d", "vocab", "ctx", "n_layers"] {
        if posint(k).is_none() {
            v.e("3", "3.3", format!("model.{} is not a positive integer", k));
        }
    }
    let block = doc.get("block");
    let repeat = block.and_then(|b| b.get("repeat")).and_then(|x| x.as_i64()).unwrap_or(0);
    if posint("n_layers") != Some(repeat) {
        v.e("3", "3.3", format!("block.repeat {} != model.n_layers {:?}", repeat, posint("n_layers")));
    }
    let l = repeat.max(0) as usize;
    let embed = nodes(doc, "embed");
    let blk = nodes(doc, "block");
    let head = nodes(doc, "head");
    let slots: Vec<&Json> = block.and_then(|b| b.get("state")).and_then(|s| s.as_arr()).map(|a| a.iter().collect()).unwrap_or_default();
    let has_wi = block.and_then(|b| b.get("weight_index")).is_some();

    reads(doc, v);

    // per-node rules
    for (sec, list) in [("embed", &embed), ("block", &blk), ("head", &head)] {
        let in_block = sec == "block";
        for (ni, n) in list.iter().enumerate() {
            let o = op(n);
            let where_ = format!("{}.nodes[{}] ({})", sec, ni, o);
            if let Json::Obj(m) = n {
                for (k, val) in m {
                    if is_per_layer(val) {
                        if !in_block {
                            v.e("6", "3.8", format!("{}: per_layer {} outside the block", where_, k));
                        } else if NODE_STRUCTURAL.contains(&k.as_str()) {
                            v.e("6", "3.8", format!("{}: {} is structural and cannot be per_layer", where_, k));
                        } else {
                            let len = val.get("per_layer").and_then(|p| p.as_arr()).map(|a| a.len()).unwrap_or(0);
                            if len != l {
                                v.e("6", "3.8", format!("{}: per_layer {} has {} entries, block.repeat is {}", where_, k, len, l));
                            }
                        }
                    }
                }
            }
            // rule 5 closed sets, per application
            for (cop, key, allowed) in CLOSED {
                if o != cop {
                    continue;
                }
                if let Some(val) = n.get(key) {
                    let vals: Vec<&Json> = if is_per_layer(val) {
                        val.get("per_layer").and_then(|p| p.as_arr()).map(|a| a.iter().collect()).unwrap_or_default()
                    } else {
                        vec![val]
                    };
                    for x in vals {
                        if !x.as_str().map(|s| allowed.contains(&s)).unwrap_or(false) {
                            v.e("5", "5", format!("{}: {} = {} is outside {:?}", where_, key, crate::json::to_string(x), allowed));
                        }
                    }
                }
            }
            // per_layer element types
            if in_block {
                if let Json::Obj(m) = n {
                    for (k, val) in m {
                        if let Some(Json::Arr(items)) = val.get("per_layer") {
                            if !is_per_layer(val) {
                                continue;
                            }
                            for x in items {
                                let ok = if INT_ATTRS.contains(&k.as_str()) {
                                    x.as_i64().map(|i| i >= 0).unwrap_or(false)
                                } else if NUM_ATTRS.contains(&k.as_str()) {
                                    matches!(x, Json::Num(_))
                                } else if BOOL_ATTRS.contains(&k.as_str()) {
                                    matches!(x, Json::Bool(_))
                                } else {
                                    true
                                };
                                if !ok && k != "step" {
                                    v.e("1", "3.8", format!("{}: per_layer {} holds {}", where_, k, crate::json::to_string(x)));
                                }
                            }
                        }
                    }
                }
            }
            // rule 7 arity
            if let Some((lo, hi, no)) = arity(o) {
                let ni_ = ins(n).len();
                if ni_ < lo || ni_ > hi {
                    v.e("7", "5", format!("{}: reads {} value(s)", where_, ni_));
                }
                let outs_ = outs(n);
                match no {
                    0 => {}
                    1 => {
                        if !matches!(n.get("out"), Some(Json::Str(_))) {
                            v.e("7", "5", format!("{}: writes {} names, not one", where_, outs_.len()));
                        }
                    }
                    k => {
                        if outs_.len() != k {
                            v.e("7", "5", format!("{}: writes {} names, not {}", where_, outs_.len(), k));
                        }
                    }
                }
            }
            if o == "depth_route" && n.get("out").is_none() {
                v.e("20", "7.4", format!("{}: no out", where_));
            }
            // stateful ops outside the block (rule 11), depth_route outside (rule 20)
            if !in_block && STATEFUL.contains(&o) {
                v.e("11", "6.1", format!("{}: a stateful op outside the block", where_));
            }
            if o == "depth_route" && (!in_block || ni != 0) {
                v.e("20", "7.4", format!("{}: depth_route must be the block's first node", where_));
            }
            // {i}
            for (k, name) in tensor_names(n) {
                let c = name.matches("{i}").count();
                if c > 1 {
                    v.e("17a", "3.7", format!("{}: {} {:?} holds {{i}} more than once", where_, k, name));
                }
                if c > 0 && !in_block {
                    v.e("17a", "3.7", format!("{}: {} {:?} holds {{i}} outside the block", where_, k, name));
                }
            }
            // tokens and pos
            let inputs = ins(n);
            for (pi, nm) in inputs.iter().enumerate() {
                if nm == "tokens" && o != "embedding" {
                    v.e("17a", "3.5", format!("{}: reads tokens", where_));
                }
                if nm == "pos" && !((o == "rope" && pi == 2) || (o == "pos_embedding" && pi == 1)) {
                    v.e("17a", "3.5", format!("{}: reads pos other than as rope's third or pos_embedding's second input", where_));
                }
            }
            for nm in outs(n) {
                if nm == "tokens" || nm == "pos" {
                    v.e("17a", "3.5", format!("{}: writes {}", where_, nm));
                }
            }
            // rule 9 expr
            if o == "expr" {
                if let Some(e) = n.get("expr").and_then(|x| x.as_str()) {
                    if let Err(er) = crate::expr::parse_expr(e, inputs.len()) {
                        v.e("9", "5.19", format!("{}: the expression does not parse: {}", where_, er));
                    }
                }
            }
            // state_weight placement (rule 17)
            if let Some(sw) = n.get("state_weight") {
                if sec != "embed" {
                    v.e("17", "6.8.1", format!("{}: state_weight outside the embed section", where_));
                }
                if sw.as_str().map(|s| s.contains("{i}")).unwrap_or(false) {
                    v.e("17", "6.8.1", format!("{}: state_weight holds {{i}}", where_));
                }
            }
            // rules 39 and 40 (IR_SPEC 5.15; revisions 1.3 and 1.4): the attributes' structure
            if o == "attention" {
                structure_39_40(n, &where_, v);
            }
            // rule 19
            if n.get("tensor_index").is_some() && !(in_block && has_wi) {
                v.e("19", "7.2", format!("{}: tensor_index in a block without weight_index", where_));
            }
            // rule 10 and 14, per application
            let apps = if in_block { l.max(1) } else { 1 };
            for a in 0..apps {
                let gi = |k: &str| at(n, k, a).and_then(|x| x.as_i64());
                match o {
                    "split" => {
                        let sz = n.get("sizes").and_then(|s| s.as_arr()).map(|s| s.len()).unwrap_or(0);
                        if sz != outs(n).len() {
                            v.e("10", "5.5", format!("{}: {} sizes for {} outputs", where_, sz, outs(n).len()));
                        }
                    }
                    "rmsnorm" => {
                        if let Some(g) = at(n, "group", a) {
                            if g.as_i64().map(|g| g < 1).unwrap_or(true) {
                                v.e("10", "5.6", format!("{}: group is not positive", where_));
                            }
                        }
                    }
                    "rope" => {
                        let hd = gi("head_dim").unwrap_or(0);
                        if hd % 2 != 0 {
                            v.e("10", "5.14", format!("{}: head_dim {} is odd", where_, hd));
                        }
                        if let Some(nr) = at(n, "n_rot", a) {
                            let r = nr.as_i64().unwrap_or(-1);
                            if r % 2 != 0 || r < 1 || r > hd {
                                v.e("10", "5.14", format!("{}: n_rot {} is not even and in 1 .. head_dim", where_, r));
                            }
                        }
                        if let Some(fs) = at(n, "freq_scale", a) {
                            if fs.as_f64().map(|f| f <= 0.0).unwrap_or(true) {
                                v.e("10", "5.14", format!("{}: freq_scale is not positive", where_));
                            }
                        }
                    }
                    "attention" => {
                        let (h, kv) = (gi("heads").unwrap_or(0), gi("kv_heads").unwrap_or(0));
                        if kv < 1 || h % kv != 0 {
                            v.e("10", "5.15", format!("{}: heads {} is not a multiple of kv_heads {}", where_, h, kv));
                        }
                        if gi("window").map(|w| w < 0).unwrap_or(false) || gi("sinks").map(|w| w < 0).unwrap_or(false) {
                            v.e("10", "5.15", format!("{}: negative window or sinks", where_));
                        }
                        if at(n, "causal", a).and_then(|c| c.as_bool()) != Some(true) {
                            v.e("14", "5.15", format!("{}: causal is not true", where_));
                        }
                    }
                    "linear_attention" => {
                        if at(n, "causal", a).and_then(|c| c.as_bool()) != Some(true) {
                            v.e("14", "6.4", format!("{}: causal is not true", where_));
                        }
                    }
                    "moe_ffn" => {
                        let (k, e) = (gi("top_k").unwrap_or(0), gi("n_experts").unwrap_or(0));
                        if k < 1 || k > e {
                            v.e("10", "5.16", format!("{}: top_k {} is not in 1 .. n_experts {}", where_, k, e));
                        }
                    }
                    "conv1d_causal" => {
                        if gi("channels").unwrap_or(0) < 1 || gi("kernel").unwrap_or(0) < 2 {
                            v.e("10", "6.6", format!("{}: channels < 1 or kernel < 2", where_));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    // state_weight on more than one embedding
    if embed.iter().filter(|n| n.get("state_weight").is_some()).count() > 1 {
        v.e("17", "6.8.1", "state_weight on more than one embedding".into());
    }
    let has_sw = embed.iter().any(|n| n.get("state_weight").is_some());

    // slots (rules 11-13, 15, 16, 17a, 21, 22)
    let mut names: Vec<String> = Vec::new();
    for (si, s) in slots.iter().enumerate() {
        let name = s.get("name").and_then(|x| x.as_str()).unwrap_or("");
        let kind = s.get("kind").and_then(|x| x.as_str()).unwrap_or("");
        if names.iter().any(|x| x == name) {
            v.e("11", "6.1", format!("slot {} is declared twice", name));
        }
        names.push(name.to_string());
        if kind != "kv" && kind != "recurrent" {
            v.e("11", "6.1", format!("slot {}: kind {:?} is not kv or recurrent", name, kind));
        }
        if let Json::Obj(m) = s {
            for (k, val) in m {
                if is_per_layer(val) {
                    if SLOT_STRUCTURAL.contains(&k.as_str()) {
                        v.e("6", "3.8", format!("slot {}: {} is structural and cannot be per_layer", name, k));
                    } else if k != "kv_from" {
                        let len = val.get("per_layer").and_then(|p| p.as_arr()).map(|a| a.len()).unwrap_or(0);
                        if len != l {
                            v.e("6", "3.8", format!("slot {}: per_layer {} has {} entries", name, k, len));
                        }
                    }
                }
            }
        }
        let readers: Vec<(usize, &Json)> =
            blk.iter().enumerate().filter(|(_, n)| n.get("state").and_then(|x| x.as_str()) == Some(name)).map(|(i, n)| (i, *n)).collect();
        if readers.is_empty() {
            v.e("11", "6.1", format!("slot {} is read by no node", name));
        }
        if kind == "kv" {
            for k in ["heads", "kv_heads", "head_dim", "max_len", "window", "sinks"] {
                if s.get(k).is_none() {
                    v.e("12", "6.2", format!("kv slot {} lacks {}", name, k));
                }
            }
            if s.get("kv_from").is_some() {
                v.e("22", "7.6", format!("kv slot {} carries kv_from (reserved)", name));
            }
            let att: Vec<&&Json> = readers.iter().map(|(_, n)| n).filter(|n| op(n) == "attention").collect();
            if att.len() > 1 {
                v.e("17a", "6.1", format!("kv slot {} is read by {} attention nodes", name, att.len()));
            }
            let evict = s.get("evict").and_then(|x| x.as_str()) == Some("turn_state");
            for a in 0..l.max(1) {
                let sw = at(s, "window", a).and_then(|x| x.as_i64()).unwrap_or(0);
                let ss = at(s, "sinks", a).and_then(|x| x.as_i64()).unwrap_or(0);
                for n in &att {
                    for k in ["window", "sinks", "kv_heads", "head_dim"] {
                        let nv = at(n, k, a).and_then(|x| x.as_i64());
                        let sv = at(s, k, a).and_then(|x| x.as_i64());
                        if nv != sv {
                            v.e("12", "5.15", format!("attention reading {}: {} {:?} differs from the slot's {:?} (application {})", name, k, nv, sv, a));
                        }
                    }
                    let mask = at(n, "mask", a).and_then(|x| x.as_str()).unwrap_or("causal");
                    if mask == "turn_state" {
                        let nw = at(n, "window", a).and_then(|x| x.as_i64()).unwrap_or(0);
                        let ns = at(n, "sinks", a).and_then(|x| x.as_i64()).unwrap_or(0);
                        if nw != 0 || ns != 0 || sw != 0 || ss != 0 {
                            v.e("15", "6.8.2", format!("mask turn_state with a window or sinks (application {})", a));
                        }
                        if !has_sw {
                            v.e("15", "6.8.2", "mask turn_state without an embedding state_weight".into());
                        }
                    } else if evict {
                        v.e("16", "6.8.3", format!("evict turn_state on {} with an attention node whose mask is {}", name, mask));
                    }
                }
            }
            for (_, n) in &readers {
                if op(n) != "attention" {
                    v.e("11", "6.1", format!("{} names kv slot {}", op(n), name));
                }
            }
        } else if kind == "recurrent" {
            let sop = s.get("op").and_then(|x| x.as_str()).unwrap_or("");
            if s.get("dtype").and_then(|x| x.as_str()) != Some("f32") {
                v.e("13", "6.3", format!("recurrent slot {}: dtype is not f32", name));
            }
            let tensors: Vec<&Json> = s.get("tensors").and_then(|t| t.as_arr()).map(|a| a.iter().collect()).unwrap_or_default();
            if tensors.is_empty() {
                v.e("13", "6.3", format!("recurrent slot {}: no tensors", name));
            }
            let tshape = |t: &Json| -> Vec<i64> {
                t.get("shape").and_then(|s| s.as_arr()).map(|a| a.iter().map(|x| x.as_i64().unwrap_or(0)).collect()).unwrap_or_default()
            };
            for t in &tensors {
                if tshape(t).iter().any(|&x| x < 1) {
                    v.e("13", "6.3", format!("recurrent slot {}: a non-positive shape", name));
                }
                let init = t.get("init");
                let init_ok = match init {
                    Some(Json::Num(x)) => x.v == 0.0,
                    Some(Json::Str(s)) => s == "body" && sop == "recurrence",
                    _ => false,
                };
                if !init_ok {
                    v.e("13", "6.3", format!("recurrent slot {}: tensor init is not 0", name));
                }
            }
            if sop == "recurrence" {
                let recs: Vec<&&Json> = readers.iter().map(|(_, n)| n).filter(|n| op(n) == "recurrence").collect();
                if readers.len() > 1 {
                    v.e("33", "6.10", format!("recurrent slot {} of op recurrence is read by {} nodes", name, readers.len()));
                }
                let _ = recs;
            }
            for (_, n) in &readers {
                let o = op(n);
                if o == "attention" {
                    v.e("11", "6.1", format!("attention names recurrent slot {}", name));
                    continue;
                }
                if o != sop {
                    v.e("13", "6.1", format!("recurrent slot {} has op {} but is read by {}", name, sop, o));
                    continue;
                }
                let tn = |t: &Json| t.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string();
                for a in 0..l.max(1) {
                    let gi = |k: &str| at(n, k, a).and_then(|x| x.as_i64()).unwrap_or(-1);
                    let want: Option<Vec<(&str, Vec<i64>)>> = match o {
                        "linear_attention" => Some(vec![("num", vec![gi("heads"), gi("head_dim")]), ("den", vec![gi("heads"), gi("head_dim")])]),
                        "ssm_scan" => Some(vec![("s", vec![gi("n")])]),
                        "conv1d_causal" => Some(vec![("x", vec![gi("kernel") - 1, gi("channels")])]),
                        "selective_scan" => Some(vec![("s", vec![gi("channels"), gi("d_state")])]),
                        _ => None,
                    };
                    if let Some(w) = want {
                        let have: Vec<(String, Vec<i64>)> = tensors.iter().map(|t| (tn(t), tshape(t))).collect();
                        let ok = have.len() == w.len() && w.iter().all(|(wn, ws)| have.iter().any(|(hn, hs)| hn == wn && hs == ws));
                        if !ok {
                            v.e("13", "6.3", format!("recurrent slot {}: tensors {:?} do not fit {} ({:?})", name, have, o, w));
                            break;
                        }
                    }
                }
            }
        }
        let _ = si;
    }
    // every stateful node names a declared slot of its kind
    for (ni, n) in blk.iter().enumerate() {
        let o = op(n);
        if !STATEFUL.contains(&o) {
            continue;
        }
        let sname = n.get("state").and_then(|x| x.as_str()).unwrap_or("");
        match slots.iter().find(|s| s.get("name").and_then(|x| x.as_str()) == Some(sname)) {
            None => v.e("11", "6.1", format!("block.nodes[{}] ({}) names no declared slot ({:?})", ni, o, sname)),
            Some(s) => {
                let kind = s.get("kind").and_then(|x| x.as_str()).unwrap_or("");
                let need = if o == "attention" { "kv" } else { "recurrent" };
                if kind != need {
                    v.e("11", "6.1", format!("block.nodes[{}] ({}) names slot {} of kind {}", ni, o, sname, kind));
                }
            }
        }
    }
    // recurrence nodes (rules 31-33)
    for (ni, n) in blk.iter().enumerate() {
        if op(n) != "recurrence" {
            continue;
        }
        match Rec::compile(n) {
            Err(e) => v.errors.push(e),
            Ok(rec) => {
                let sname = n.get("state").and_then(|x| x.as_str()).unwrap_or("");
                if let Some(s) = slots.iter().find(|s| s.get("name").and_then(|x| x.as_str()) == Some(sname)) {
                    if s.get("op").and_then(|x| x.as_str()) != Some("recurrence") {
                        v.e("33", "6.10", format!("block.nodes[{}]: slot {} is not of op recurrence", ni, sname));
                    }
                    let tensors: Vec<&Json> = s.get("tensors").and_then(|t| t.as_arr()).map(|a| a.iter().collect()).unwrap_or_default();
                    let want = rec.state_names();
                    let ok = tensors.len() == want.len()
                        && want.iter().all(|(wn, ws, body)| {
                            tensors.iter().any(|t| {
                                t.get("name").and_then(|x| x.as_str()) == Some(wn.as_str())
                                    && t.get("shape").and_then(|x| x.as_arr()).map(|a| a.iter().map(|x| x.as_i64().unwrap_or(0) as usize).collect::<Vec<_>>())
                                        == Some(ws.clone())
                                    && (t.get("init").and_then(|x| x.as_str()) == Some("body")) == *body
                            })
                        });
                    if !ok {
                        v.e("33", "6.10", format!("block.nodes[{}]: slot {}'s tensors are not the update's {:?}", ni, sname, want));
                    }
                }
            }
        }
    }
    // embed or head recurrence nodes are already refused as stateful; compile them for messages anyway
    // depth_route (rules 20, 21, 23), weight_index (18)
    let drs: Vec<usize> = blk.iter().enumerate().filter(|(_, n)| op(n) == "depth_route").map(|(i, _)| i).collect();
    if drs.len() > 1 {
        v.e("20", "7.4", "depth_route occurs more than once".into());
    }
    if let Some(&di) = drs.first() {
        let n = blk[di];
        let r = n.get("n_rec").and_then(|x| x.as_i64()).unwrap_or(0);
        if r < 1 {
            v.e("20", "7.4", "n_rec is not an integer >= 1".into());
        }
        if n.get("route").and_then(|x| x.as_str()) != Some("token_choice") {
            v.e("20", "7.7", "route is not token_choice".into());
        }
        if !matches!(n.get("gate").and_then(|x| x.as_str()), Some("score") | Some("none")) {
            v.e("20", "7.4", "gate is not score or none".into());
        }
        if n.get("weight").and_then(|x| x.as_str()).map(|w| w.contains('{')).unwrap_or(true) {
            v.e("20", "7.7", "the router's weight name holds '{'".into());
        }
        let steps: Option<Vec<i64>> =
            n.get("step").and_then(|s| s.get("per_layer")).and_then(|p| p.as_arr()).map(|a| a.iter().map(|x| x.as_i64().unwrap_or(-2)).collect());
        match steps {
            None => v.e("20", "7.4", "step is not a per_layer list".into()),
            Some(st) => {
                let mut ok = st.len() == l && st.iter().all(|&x| x >= -1 && x < r);
                if ok {
                    // consecutive, ascending 0 .. R-1, equal lengths
                    let mut runs: Vec<(i64, usize)> = Vec::new();
                    for &x in &st {
                        if x < 0 {
                            continue;
                        }
                        match runs.last_mut() {
                            Some((y, c)) if *y == x => *c += 1,
                            _ => runs.push((x, 1)),
                        }
                    }
                    let in_order = runs.iter().enumerate().all(|(i, (x, _))| *x == i as i64) && runs.len() as i64 == r;
                    let equal = runs.windows(2).all(|w| w[0].1 == w[1].1);
                    // IR_SPEC 7.4 (1.2): the steps' applications form one run, -1 only before and after.
                    let inside: Vec<usize> = st.iter().enumerate().filter(|(_, &x)| x >= 0).map(|(i, _)| i).collect();
                    let one_run = !inside.is_empty() && inside.last().unwrap() - inside[0] + 1 == inside.len();
                    ok = in_order && equal && one_run;
                }
                if !ok {
                    v.e("20", "7.4", format!("step {:?} does not form the consecutive, ascending, equal steps 0 .. {}", st, r - 1));
                }
            }
        }
        let kvs: Vec<&&Json> = slots.iter().filter(|s| s.get("kind").and_then(|x| x.as_str()) == Some("kv")).collect();
        if kvs.is_empty() {
            v.e("21", "7.5", "depth_route with no kv slot".into());
        }
        for s in kvs {
            if s.get("kv_rows").and_then(|x| x.as_str()) != Some("active") {
                v.e("21", "7.5", "depth_route with a kv slot whose kv_rows is not active".into());
            }
        }
        let all: Vec<&&Json> = embed.iter().chain(blk.iter()).chain(head.iter()).collect();
        if all.iter().any(|n| matches!(op(n), "list" | "append" | "depth_attend")) {
            v.e("23", "7.4", "depth_route together with list values".into());
        }
        if blk.iter().any(|n| op(n) == "attention" && (0..l.max(1)).any(|a| at(n, "mask", a).and_then(|x| x.as_str()) == Some("turn_state"))) {
            v.e("23", "9.2", "depth_route together with mask turn_state".into());
        }
        // a node reads depth_route's out
        if let Some(out) = n.get("out").and_then(|x| x.as_str()) {
            for m in all.iter() {
                if ins(m).iter().any(|x| x == out) {
                    v.e("17a", "7.4", format!("{} reads depth_route's out {}", op(m), out));
                }
            }
        }
    }
    if let Some(wi) = block.and_then(|b| b.get("weight_index")) {
        let ms: Option<Vec<i64>> = wi.get("per_layer").and_then(|p| p.as_arr()).map(|a| a.iter().map(|x| x.as_i64().unwrap_or(-1)).collect());
        match ms {
            Some(ms) if ms.len() == l && ms.iter().all(|&x| x >= 0) => {
                let mx = *ms.iter().max().unwrap_or(&0);
                for k in 0..=mx {
                    if !ms.contains(&k) {
                        v.e("18", "7.1", format!("weight_index leaves stored layer {} unused", k));
                    }
                }
            }
            _ => v.e("18", "7.1", "weight_index is not per_layer of block.repeat integers >= 0".into()),
        }
    }
}

/// Rules 39 and 40 on the document (IR_SPEC 5.15, 10.1): `position_bias` is {weight, buckets}, buckets a
/// non-empty list of integers >= 0; `relative_kv` is {key, max_distance} with an optional `value`, key a
/// name, value a name or null, max_distance an integer >= 0, no other member; not both on one node. A
/// `per_layer` value of either is rule 6 (structural, 3.8) and is not judged here. `_` members are prose.
fn structure_39_40(n: &Json, where_: &str, v: &mut V) {
    let name = |x: Option<&Json>| matches!(x, Some(Json::Str(s)) if !s.is_empty());
    let extra = |x: &Json, allowed: &[&str]| -> Vec<String> {
        match x {
            Json::Obj(m) => m.iter().map(|(k, _)| k.clone()).filter(|k| !k.starts_with('_') && !allowed.contains(&k.as_str())).collect(),
            _ => vec![],
        }
    };
    if let Some(pb) = n.get("position_bias") {
        if !is_per_layer(pb) {
            let buckets_ok = match pb.get("buckets") {
                Some(Json::Arr(a)) => !a.is_empty() && a.iter().all(|x| matches!(x, Json::Num(_)) && x.as_i64().map(|b| b >= 0).unwrap_or(false)),
                _ => false,
            };
            let others = extra(pb, &["weight", "buckets"]);
            if !matches!(pb, Json::Obj(_)) || !name(pb.get("weight")) || !buckets_ok || !others.is_empty() {
                v.e("39", "5.15", format!("{}: position_bias {} is not {{weight, buckets}} with buckets a non-empty list of integers >= 0", where_, crate::json::to_string(pb)));
            }
        }
    }
    if let Some(rk) = n.get("relative_kv") {
        if !is_per_layer(rk) {
            let kd_ok = matches!(rk.get("max_distance"), Some(Json::Num(_))) && rk.get("max_distance").and_then(|x| x.as_i64()).map(|k| k >= 0).unwrap_or(false);
            let value_ok = match rk.get("value") {
                None | Some(Json::Null) => true,
                x => name(x),
            };
            let others = extra(rk, &["key", "value", "max_distance"]);
            if !matches!(rk, Json::Obj(_)) || !name(rk.get("key")) || !kd_ok || !value_ok || !others.is_empty() {
                v.e("40", "5.15", format!("{}: relative_kv {} is not {{key, max_distance}} with an optional value (a name or null), max_distance an integer >= 0", where_, crate::json::to_string(rk)));
            }
        }
        if n.get("position_bias").is_some() {
            v.e("40", "5.15", format!("{}: relative_kv and position_bias on one node", where_));
        }
    }
}

/// Rule 36 (IR_SPEC 3.5, 7.4; revision 1.2): every value a node reads is written before it (embed, the
/// block as application 0, head); the embed writes x and the head logits; under depth_route the block's
/// other nodes read x, pos and values of the same application, and the head x and the embed's values.
fn reads(doc: &Json, v: &mut V) {
    use std::collections::HashSet;
    let embed = nodes(doc, "embed");
    let blk = nodes(doc, "block");
    let head = nodes(doc, "head");
    let routed = blk.first().map(|n| op(n) == "depth_route").unwrap_or(false);
    let walk = |sec: &str, list: &[&Json], have: &mut HashSet<String>, v: &mut V| {
        for (j, n) in list.iter().enumerate() {
            for (k, x) in ins(n).iter().enumerate() {
                if !have.contains(x) {
                    v.e("36", if routed && sec != "embed" { "7.4" } else { "3.5" },
                        format!("{}[{}] {} reads {} (its input {}), which no earlier node writes", sec, j, op(n), x, k));
                }
            }
            for o in outs(n) {
                have.insert(o);
            }
        }
    };
    let mut have: HashSet<String> = ["tokens", "pos"].iter().map(|s| s.to_string()).collect();
    walk("embed", &embed[..], &mut have, v);
    if !have.contains("x") {
        v.e("36", "3.5", "the embed section writes no x".into());
    }
    if routed {
        let embed_have = have.clone();
        walk("block", &blk[..1], &mut have, v);
        let mut step: HashSet<String> = ["x", "pos"].iter().map(|s| s.to_string()).collect();
        walk("block", &blk[1..], &mut step, v);
        let mut h = embed_have;
        h.insert("x".into());
        walk("head", &head[..], &mut h, v);
    } else {
        walk("block", &blk[..], &mut have, v);
        walk("head", &head[..], &mut have, v);
    }
    if !head.iter().any(|n| outs(n).iter().any(|o| o == "logits")) {
        v.e("36", "3.4", "the head writes no logits".into());
    }
}

fn file_arch(g: &Gguf) -> String {
    g.get_str("general.architecture").unwrap_or_default()
}

/// A 2-D ne {ne0, ne1}, compared as stored with the dimension count included (IR_SPEC 3.1, v1.4.2):
/// a stored ne {w} is not ne {w, 1}.
fn ne2(ne: &[u64]) -> Option<(u64, u64)> {
    if ne.len() != 2 {
        return None;
    }
    Some((ne[0], ne[1]))
}

/// Rule 38 (IR_SPEC 3.1, 10.1, Appendix D; v1.4.2). The label is "paperkiln" or one of Appendix D's eight
/// architectures (an absent general.architecture is refused); for the eight, the embedded document must
/// equal, in D.7's normal form, the graph D.1 to D.6 build from the file's metadata and tensor names, and
/// metadata that builds no graph is refused. Judged on the embedded document always, also when another
/// is read in place of it, and never on the document read in place.
fn rule_38(g: &Gguf, v: &mut V) {
    let text = match g.get_str("paperkiln.ir") {
        Some(t) => t,
        None => return,
    };
    let arch = match g.get("general.architecture") {
        None => {
            v.e("38", "3.1", "a file with an embedded document has no general.architecture".into());
            return;
        }
        Some(a) => match a.as_str() {
            Some(s) => s,
            None => {
                v.e("38", "3.1", "general.architecture is not a string".into());
                return;
            }
        },
    };
    if arch == "paperkiln" {
        return;
    }
    if !crate::arch::ARCHS.contains(&arch.as_str()) {
        v.e("38", "3.1", format!("general.architecture {:?} is neither \"paperkiln\" nor one of Appendix D's eight", arch));
        return;
    }
    let graph = match crate::arch::build(g) {
        Ok(d) => d,
        Err(e) => {
            v.e("38", "D.1", format!("general.architecture {:?}: the metadata builds no graph ({})", arch, e));
            return;
        }
    };
    let emb = match crate::json::parse_value(&text) {
        Ok(d) => d,
        Err(e) => {
            v.e("38", "3.1", format!("the embedded document does not parse ({}), so it is not {:?}'s graph", e, arch));
            return;
        }
    };
    let has = |n: &str| g.info(n).is_some();
    let a = crate::arch::normal_form(&emb, &has);
    let b = crate::arch::normal_form(&graph, &has);
    if let Some(d) = crate::arch::first_diff(&a, &b, "$") {
        v.e("38", "D.7", format!("general.architecture {:?}, but the embedded document is not the graph Appendix D builds from the file: {}", arch, d));
    }
}

fn container(doc: &Json, g: &Gguf, v: &mut V) {
    rule_38(g, v);
    // rule 24 and 30: against the document embedded in the file
    if let Some(text) = g.get_str("paperkiln.ir") {
        let emb = crate::json::parse_value(&text);
        match (g.get("paperkiln.ir_version"), &emb) {
            (Some(k), Ok(e)) => {
                let kv = if k.is_u32() { k.as_u64() } else { None };
                let dv = e.get("ir_version").and_then(|x| x.as_i64());
                if kv.is_none() || kv.map(|x| x as i64) != dv {
                    v.e("24", "3.1", format!("paperkiln.ir_version {:?} is not a uint32 equal to the embedded document's ir_version {:?}", k, dv));
                }
            }
            (None, _) => v.e("24", "3.1", "paperkiln.ir_version is missing".into()),
            _ => {}
        }
        if let Some(hv) = g.get("paperkiln.ir_sha256") {
            match (hv.as_str(), digest::digest(&text, false)) {
                (Some(h), Ok((d, _))) if h == d => {}
                (Some(h), Ok((d, _))) => v.e("30", "3.10", format!("paperkiln.ir_sha256 {} is not the embedded document's digest {}", h, d)),
                (Some(_), Err(e)) => v.e("30", "3.10", format!("the embedded document has no digest: {}", e)),
                (None, _) => v.e("30", "3.10", "paperkiln.ir_sha256 is not a string".into()),
            }
        }
    }
    let blk = nodes(doc, "block");
    let block = doc.get("block");
    let l = block.and_then(|b| b.get("repeat")).and_then(|x| x.as_i64()).unwrap_or(0).max(0) as usize;
    let wi: Vec<usize> = match block.and_then(|b| b.get("weight_index")).and_then(|w| w.get("per_layer")).and_then(|p| p.as_arr()) {
        Some(a) => a.iter().map(|x| x.as_i64().unwrap_or(0).max(0) as usize).collect(),
        None => (0..l).collect(),
    };
    let has = |n: &str| g.info(n).is_some();
    let resolve = |node: &Json, name: &str, a: usize| -> String {
        if !name.contains("{i}") {
            return name.to_string();
        }
        let idx = if node.get("tensor_index").and_then(|x| x.as_str()) == Some("application") { a } else { wi.get(a).copied().unwrap_or(a) };
        name.replace("{i}", &idx.to_string())
    };
    // list lengths, for depth_attend's S (5.18)
    let mut lists: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let check_node = |n: &Json, a: Option<usize>, lists: &mut std::collections::HashMap<String, usize>, v: &mut V| {
        let o = op(n);
        match o {
            "list" => {
                lists.insert(outs(n).first().cloned().unwrap_or_default(), ins(n).len());
            }
            "append" => {
                let i = ins(n);
                let base = i.first().and_then(|x| lists.get(x)).copied().unwrap_or(0);
                lists.insert(outs(n).first().cloned().unwrap_or_default(), base + 1);
            }
            _ => {}
        }
        let s_count = if o == "depth_attend" {
            let i = ins(n);
            Some(i.first().and_then(|x| lists.get(x)).copied().unwrap_or(1) + i.len().saturating_sub(1))
        } else {
            None
        };
        for (k, name) in tensor_names(n) {
            if k == "tied_to" {
                continue;
            }
            let full = resolve(n, &name, a.unwrap_or(0));
            if has(&full) {
                continue;
            }
            if o == "linear" && k == "weight" {
                if let Some(t) = n.get("tied_to").and_then(|x| x.as_str()) {
                    let tf = resolve(n, t, a.unwrap_or(0));
                    if has(&tf) {
                        continue;
                    }
                    v.e("25", "3.7", format!("{}: neither {} nor its tied_to {} is a tensor of the file", o, full, tf));
                    continue;
                }
            }
            if o == "depth_attend" && k == "weight" && s_count == Some(1) {
                v.warnings.push(format!("exporter rule (IR_SPEC 5.18): pseudo-query {} is absent; S = 1 never reads it", full));
                continue;
            }
            v.e("25", "3.7", format!("{}: tensor {} is not a tensor of the file", o, full));
        }
        // rule 34: recurrence weights' element counts
        if o == "recurrence" {
            if let Ok(rec) = Rec::compile(n) {
                for (tid, tn) in &rec.weights {
                    let full = resolve(n, tn, a.unwrap_or(0));
                    if let Some(info) = g.info(&full) {
                        let cnt: u64 = info.ne.iter().product();
                        let need: usize = rec.tensors[*tid].dims.iter().product();
                        if cnt as usize != need {
                            v.e("34", "6.10", format!("weight {} holds {} elements, its declaration {}", full, cnt, need));
                        }
                    }
                }
            }
        }
    };
    for n in nodes(doc, "embed") {
        check_node(n, None, &mut lists, v);
    }
    for a in 0..l {
        for n in &blk {
            check_node(n, Some(a), &mut lists, v);
        }
    }
    for n in nodes(doc, "head") {
        check_node(n, None, &mut lists, v);
    }
    // rules 39 and 40, container (IR_SPEC 5.15): each table's ne, in every application that reads it
    for a in 0..l {
        for n in &blk {
            if op(n) != "attention" {
                continue;
            }
            let gi = |k: &str| at(n, k, a).and_then(|x| x.as_i64()).unwrap_or(0).max(0) as u64;
            let (heads, hd) = (gi("heads"), gi("head_dim"));
            if let Some(pb) = n.get("position_bias").filter(|x| !is_per_layer(x)) {
                if let Some(info) = pb.get("weight").and_then(|x| x.as_str()).and_then(|w| g.info(&resolve(n, w, a))) {
                    let maxb = pb.get("buckets").and_then(|b| b.as_arr()).map(|b| b.iter().filter_map(|x| x.as_i64()).max().unwrap_or(0)).unwrap_or(0);
                    let ne = ne2(&info.ne);
                    if !(ne.is_some() && ne.unwrap().0 == heads && ne.unwrap().1 as i64 > maxb) {
                        v.e("39", "5.15", format!("position_bias table {} has ne {:?}, not {{heads = {}, B}} with B > {} (application {})", resolve(n, pb.get("weight").and_then(|x| x.as_str()).unwrap_or(""), a), info.ne, heads, maxb, a));
                    }
                }
            }
            if let Some(rk) = n.get("relative_kv").filter(|x| !is_per_layer(x)) {
                let kd = rk.get("max_distance").and_then(|x| x.as_i64()).unwrap_or(-1);
                for key in ["key", "value"] {
                    if let Some(tn) = rk.get(key).and_then(|x| x.as_str()) {
                        let full = resolve(n, tn, a);
                        if let Some(info) = g.info(&full) {
                            let ok = match ne2(&info.ne) {
                                Some((w, rows)) => (w == hd || w == heads * hd) && rows as i64 == kd + 1,
                                None => false,
                            };
                            if !ok {
                                v.e("40", "5.15", format!("relative_kv.{} {} has ne {:?}, not {{{} or {}, max_distance + 1 = {}}} (application {})", key, full, info.ne, hd, heads * hd, kd + 1, a));
                            }
                        }
                    }
                }
            }
        }
    }
    let d = doc.get("model").and_then(|m| m.get("d")).and_then(|x| x.as_i64()).unwrap_or(0) as u64;
    // rule 26
    if let Some(dr) = blk.iter().find(|n| op(n) == "depth_route") {
        let r = dr.get("n_rec").and_then(|x| x.as_i64()).unwrap_or(0) as u64;
        if let Some(w) = dr.get("weight").and_then(|x| x.as_str()).and_then(|w| g.info(w)) {
            if w.ne != vec![d, r] {
                v.e("26", "7.4", format!("the router's ne {:?} is not {{d, n_rec}} = [{}, {}]", w.ne, d, r));
            }
        }
    }
    // rule 27
    if block.and_then(|b| b.get("weight_index")).is_some() {
        let arch = file_arch(g);
        if arch != "paperkiln" {
            v.e("27", "7.1", format!("weight_index in a file of architecture {:?}, not paperkiln", arch));
        }
        let stored = wi.iter().max().map(|m| m + 1).unwrap_or(0) as u64;
        match g.get(&format!("{}.block_count", arch)).and_then(|x| x.as_u64()) {
            None => v.e("27", "7.1", format!("{}.block_count is missing", arch)),
            Some(bc) if bc != stored => v.e("27", "7.1", format!("{}.block_count {} differs from the {} stored layers", arch, bc, stored)),
            _ => {}
        }
    }
    // rule 28: state tokens
    let k = g.get("paperkiln.state_tokens.k").and_then(|x| x.as_u64()).unwrap_or(0);
    let sw: Option<String> = nodes(doc, "embed").iter().find_map(|n| n.get("state_weight").and_then(|x| x.as_str()).map(String::from));
    if k > 0 || sw.is_some() {
        if k == 0 || sw.is_none() {
            v.e("28", "6.8.1", "paperkiln.state_tokens.k without an embedding state_weight, or the reverse".into());
        }
        if let Some(swn) = &sw {
            match g.info(swn) {
                Some(info) => {
                    if info.ne != vec![d, k] {
                        v.e("28", "6.8.1", format!("state_weight's ne {:?} is not {{d, k}} = [{}, {}]", info.ne, d, k));
                    }
                }
                None => v.e("28", "6.8.1", format!("state_weight {} is not a tensor of the file", swn)),
            }
        }
        if file_arch(g) != "paperkiln" {
            v.e("28", "8.2", format!("state tokens in a file of architecture {:?}, not paperkiln", file_arch(g)));
        }
        let specials = doc.get("tokenizer").and_then(|t| t.get("specials"));
        let toks: Option<Vec<String>> = match g.get("tokenizer.ggml.tokens") {
            Some(crate::gguf::Val::Arr(a)) => Some(a.iter().map(|x| x.as_str().unwrap_or_default()).collect()),
            _ => None,
        };
        for (role, marker) in [("eot", "<|endoftext|>"), ("user", "<|user|>"), ("assistant", "<|assistant|>")] {
            match specials.and_then(|s| s.get(role)).and_then(|x| x.as_i64()) {
                None => v.e("28", "8.2", format!("tokenizer.specials lacks {}", role)),
                Some(id) => {
                    let tok = toks.as_ref().and_then(|t| t.get(id as usize)).cloned();
                    if tok.as_deref() != Some(marker) {
                        v.e("28", "8.2", format!("tokenizer.specials.{} = {} names {:?}, not {}", role, id, tok, marker));
                    }
                }
            }
        }
        if let Some(ct) = g.get_str("paperkiln.chat_template") {
            let ok = crate::json::parse_value(&ct)
                .ok()
                .map(|r| r.get("name").and_then(|x| x.as_str()) == Some("paperkiln") && r.get("version").and_then(|x| x.as_i64()) == Some(1))
                .unwrap_or(false);
            if !ok {
                v.e("28", "8.2", "a declared paperkiln.chat_template other than paperkiln/1 with state tokens".into());
            }
        }
    }
}
