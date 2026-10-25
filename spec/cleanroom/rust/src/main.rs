//! kcr: a clean-room engine for the keramos IR, speaking the conformance adapter contract
//! (IR_SPEC 11.3) as JSON Lines on stdin and stdout.
//!
//!     kcr --conformance            the adapter
//!     kcr check FILE.gguf [DOC]    validate a file (and a document in place of its own)
//!     kcr digest FILE              the digest of a JSON document

mod digest;
mod expr;
mod gguf;
mod json;
mod model;
mod ops;
mod quant;
mod recurrence;
mod schema;
mod session;
mod sha256;
mod tensor;
mod validate;

use json::Json;
use model::{op, outs, Model};
use ops::{Slots, Val};
use session::Session;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use tensor::{tensor_from_json, tensor_to_json, Tensor};

const ENGINE: &str = "keramos-cleanroom (Rust)";
const PROFILES: [&str; 5] = ["core", "state", "recursion", "thirdparty", "recurrence"];
const METHODS: [&str; 11] =
    ["hello", "check", "open", "forward", "truncate", "snapshot", "restore", "close", "run_node", "digest", "dequantize"];

struct Loaded {
    model: Model,
    sess: Session,
}

struct Engine {
    sessions: HashMap<String, Loaded>,
    next: usize,
}

enum Reply {
    Result(Json),
    Error(String),
    Skip(String),
}

fn p_str<'a>(p: &'a Json, k: &str) -> Option<&'a str> {
    p.get(k).and_then(|v| v.as_str())
}

/// Loads a GGUF and its document (the embedded one, or `ir` in its place), refusing at load.
fn load(gguf_path: &str, ir: Option<&str>) -> Result<(gguf::Gguf, Json, Vec<String>), String> {
    let g = gguf::Gguf::open(gguf_path)?;
    let text = match ir {
        Some(p) => std::fs::read_to_string(p).map_err(|e| format!("cannot read {}: {}", p, e))?,
        None => match g.get_str("paperkiln.ir") {
            Some(t) => t,
            None => {
                return Err(format!(
                    "{} has no paperkiln.ir; building a document from llama.cpp metadata (IR_SPEC 9.3) is not implemented",
                    gguf_path
                ))
            }
        },
    };
    let doc = json::parse_value(&text).map_err(|e| format!("IR_SPEC 10.1 (rule 1): the document is not JSON: {}", e))?;
    let rep = validate::validate(&doc, Some(&g));
    if !rep.errors.is_empty() {
        return Err(rep.errors.join("; "));
    }
    Ok((g, doc, rep.warnings))
}

fn build_model(g: &gguf::Gguf, doc: &Json) -> Result<Model, String> {
    let mut weights = HashMap::new();
    for t in &g.tensors {
        weights.insert(t.name.clone(), g.load(&t.name)?);
    }
    let k = g.get("paperkiln.state_tokens.k").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    Model::from_doc(&json::strip_prose(doc), weights, k)
}

/// The tensor names of a recurrent op's state, in the order the engine keeps them.
fn state_names(m: &Model, node: &Json, section: u8, index: usize) -> Vec<String> {
    match op(node) {
        "linear_attention" => vec!["num".into(), "den".into()],
        "ssm_scan" | "selective_scan" => vec!["s".into()],
        "conv1d_causal" => vec!["x".into()],
        "recurrence" => m.recs.get(&(section, index)).map(|r| r.state_names().into_iter().map(|x| x.0).collect()).unwrap_or_default(),
        _ => vec![],
    }
}

fn initial_rec_state(m: &Model, node: &Json, ni: usize, a: usize, slot: &Json) -> Result<Vec<Tensor>, String> {
    if op(node) == "recurrence" {
        let rec = m.recs.get(&(1, ni)).ok_or("recurrence not compiled")?;
        let mut wts = Vec::new();
        for (_, tn) in &rec.weights {
            wts.push(m.w(&m.tname(node, tn, Some(a)))?.data.clone());
        }
        let init = rec.initial_state(&wts);
        return Ok(rec.state.iter().zip(init).map(|(&tid, v)| Tensor::new(rec.tensors[tid].dims.clone(), v)).collect());
    }
    // hand-written ops start at +0, in the slot's shapes
    let decl: Vec<(String, Vec<usize>)> = slot_tensors(slot);
    let names = state_names(m, node, 1, ni);
    Ok(names
        .iter()
        .map(|n| Tensor::zeros(decl.iter().find(|(dn, _)| dn == n).map(|x| x.1.clone()).unwrap_or_default()))
        .collect())
}

fn slot_tensors(slot: &Json) -> Vec<(String, Vec<usize>)> {
    slot.get("tensors")
        .and_then(|t| t.as_arr())
        .map(|a| {
            a.iter()
                .map(|t| {
                    (
                        p_str(t, "name").unwrap_or("").to_string(),
                        t.get("shape").and_then(|s| s.as_arr()).map(|s| s.iter().map(|x| x.as_i64().unwrap_or(0) as usize).collect()).unwrap_or_default(),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

fn reports(m: &Model, s: &Session, want: &[String], out: Option<&session::CallOut>) -> Result<Vec<(String, Json)>, String> {
    let mut r = Vec::new();
    for w in want {
        match w.as_str() {
            "logits" => {
                let o = out.ok_or("logits are reported by forward only")?;
                r.push(("logits".into(), tensor_to_json(&o.logits)));
            }
            "depth" => {
                let o = out.ok_or("depth is reported by forward only")?;
                r.push(("depth".into(), Json::Arr(o.depth.iter().map(|&d| Json::int(d as i64)).collect())));
            }
            "kv_rows" | "kv_positions" => {
                let mut obj = Vec::new();
                for sl in &m.slots {
                    if p_str(sl, "kind") != Some("kv") {
                        continue;
                    }
                    let name = p_str(sl, "name").unwrap_or("").to_string();
                    let per: Vec<Json> = (0..m.repeat)
                        .map(|a| {
                            let pos = s.slots.kv.get(&(name.clone(), a)).map(|i| i.pos.clone()).unwrap_or_default();
                            if w == "kv_rows" {
                                Json::int(pos.len() as i64)
                            } else {
                                Json::Arr(pos.iter().map(|&p| Json::int(p as i64)).collect())
                            }
                        })
                        .collect();
                    obj.push((name, Json::Arr(per)));
                }
                r.push((w.clone(), Json::Obj(obj)));
            }
            "recurrent" => {
                let mut obj = Vec::new();
                for sl in &m.slots {
                    if p_str(sl, "kind") != Some("recurrent") {
                        continue;
                    }
                    let name = p_str(sl, "name").unwrap_or("").to_string();
                    let (ni, node) = m
                        .block
                        .iter()
                        .enumerate()
                        .find(|(_, n)| p_str(n, "state") == Some(name.as_str()))
                        .ok_or(format!("slot {} is read by no node", name))?;
                    let names = state_names(m, node, 1, ni);
                    let decl = slot_tensors(sl);
                    let mut per = Vec::new();
                    for a in 0..m.repeat {
                        let ts = match s.slots.rec.get(&(name.clone(), a)) {
                            Some(t) => t.clone(),
                            None => initial_rec_state(m, node, ni, a, sl)?,
                        };
                        let mut tobj = Vec::new();
                        for (dn, shape) in &decl {
                            let k = names.iter().position(|n| n == dn).ok_or(format!("slot {}: tensor {} unknown to its op", name, dn))?;
                            let t = Tensor::new(shape.clone(), ts[k].data.clone());
                            tobj.push((dn.clone(), tensor_to_json(&t)));
                        }
                        per.push(Json::Obj(tobj));
                    }
                    obj.push((name, Json::Arr(per)));
                }
                r.push(("recurrent".into(), Json::Obj(obj)));
            }
            other => return Err(format!("unknown report {}", other)),
        }
    }
    Ok(r)
}

fn report_list(p: &Json) -> Vec<String> {
    p.get("report").and_then(|r| r.as_arr()).map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default()
}

impl Engine {
    fn handle(&mut self, method: &str, p: &Json) -> Reply {
        match self.handle_inner(method, p) {
            Ok(r) => r,
            Err(e) => Reply::Error(e),
        }
    }

    fn sess(&mut self, p: &Json) -> Result<&mut Loaded, String> {
        let id = p_str(p, "session").ok_or("no session")?.to_string();
        self.sessions.get_mut(&id).ok_or(format!("no session {}", id))
    }

    fn handle_inner(&mut self, method: &str, p: &Json) -> Result<Reply, String> {
        Ok(match method {
            "hello" => Reply::Result(Json::obj(vec![
                ("engine", Json::str(ENGINE)),
                ("version", Json::str(concat!("0.1.0 (IR versions 0 and 1; spec revision 1.1, keramos v1.1.4)"))),
                ("backend", Json::str("cpu")),
                ("exact", Json::Bool(true)),
                ("profiles", Json::Arr(PROFILES.iter().map(|s| Json::str(s)).collect())),
                ("methods", Json::Arr(METHODS.iter().map(|s| Json::str(s)).collect())),
            ])),
            "check" => {
                let gp = p_str(p, "gguf").ok_or("no gguf")?;
                let ir = p_str(p, "ir");
                let res = load(gp, ir).and_then(|(g, doc, _)| build_model(&g, &doc).map(|_| ()));
                match res {
                    Ok(()) => Reply::Result(Json::obj(vec![("accepted", Json::Bool(true))])),
                    Err(e) => Reply::Result(Json::obj(vec![("accepted", Json::Bool(false)), ("message", Json::Str(e))])),
                }
            }
            "open" => {
                let gp = p_str(p, "gguf").ok_or("no gguf")?;
                let (g, doc, _) = load(gp, p_str(p, "ir"))?;
                let model = build_model(&g, &doc)?;
                let declared = session::evict_declared(&model);
                let evict = match p.get("evict") {
                    Some(Json::Bool(true)) => {
                        if !declared {
                            return Err("evict true on a document without evict: turn_state".into());
                        }
                        true
                    }
                    Some(Json::Bool(false)) => false,
                    _ => declared,
                };
                self.next += 1;
                let id = format!("s{}", self.next);
                let res = Json::obj(vec![
                    ("session", Json::Str(id.clone())),
                    ("vocab", Json::int(model.vocab as i64)),
                    ("applications", Json::int(model.repeat as i64)),
                    ("state_k", Json::int(model.state_k as i64)),
                ]);
                self.sessions.insert(id, Loaded { model, sess: Session::new(evict) });
                Reply::Result(res)
            }
            "forward" => {
                let ids: Vec<i64> = p.get("ids").and_then(|x| x.as_arr()).ok_or("no ids")?.iter().map(|x| x.as_i64().unwrap_or(-1)).collect();
                let want = report_list(p);
                let l = self.sess(p)?;
                let out = l.sess.forward(&l.model, &ids)?;
                let mut res = vec![("len".to_string(), Json::int(l.sess.ids.len() as i64))];
                res.extend(reports(&l.model, &l.sess, &want, Some(&out))?);
                Reply::Result(Json::Obj(res))
            }
            "truncate" => {
                let n = p.get("n").and_then(|x| x.as_i64()).ok_or("no n")? as usize;
                let want = report_list(p);
                let l = self.sess(p)?;
                l.sess.truncate(&l.model, n)?;
                let mut res = vec![("len".to_string(), Json::int(l.sess.ids.len() as i64))];
                res.extend(reports(&l.model, &l.sess, &want, None)?);
                Reply::Result(Json::Obj(res))
            }
            "snapshot" => {
                let tag = p_str(p, "tag").ok_or("no tag")?.to_string();
                let l = self.sess(p)?;
                let snap = (l.sess.ids.clone(), l.sess.slots.clone());
                l.sess.snapshots.insert(tag, snap);
                Reply::Result(Json::obj(vec![("len", Json::int(l.sess.ids.len() as i64))]))
            }
            "restore" => {
                let tag = p_str(p, "tag").ok_or("no tag")?.to_string();
                let want = report_list(p);
                let l = self.sess(p)?;
                let (ids, slots) = l.sess.snapshots.get(&tag).cloned().ok_or(format!("no snapshot {}", tag))?;
                l.sess.ids = ids;
                l.sess.slots = slots;
                let mut res = vec![("len".to_string(), Json::int(l.sess.ids.len() as i64))];
                res.extend(reports(&l.model, &l.sess, &want, None)?);
                Reply::Result(Json::Obj(res))
            }
            "close" => {
                let id = p_str(p, "session").ok_or("no session")?.to_string();
                self.sessions.remove(&id);
                Reply::Result(Json::Obj(vec![]))
            }
            "run_node" => run_node(p)?,
            "run_node_grad" => Reply::Skip("gradients are not implemented (this engine does not train)".into()),
            "digest" => {
                let text = p_str(p, "text").ok_or("no text")?;
                let prose = p.get("prose").and_then(|x| x.as_bool()).unwrap_or(false);
                match digest::digest(text, prose) {
                    Ok((d, c)) => Reply::Result(Json::obj(vec![("sha256", Json::Str(d)), ("canonical", Json::Str(c))])),
                    Err(e) => Reply::Result(Json::obj(vec![("refused", Json::Str(e))])),
                }
            }
            "dequantize" => {
                let ty = quant::type_id(p_str(p, "type").ok_or("no type")?).ok_or("storage type not defined by IR_SPEC 4.7")?;
                let n = p.get("n").and_then(|x| x.as_i64()).ok_or("no n")? as usize;
                let bytes = tensor::decode_hex_bytes(p_str(p, "bytes").ok_or("no bytes")?)?;
                let v = quant::dequantize(ty, &bytes, n)?;
                Reply::Result(Json::obj(vec![("values", tensor_to_json(&Tensor::new(vec![n], v)))]))
            }
            m => Reply::Skip(format!("method {} is not implemented", m)),
        })
    }
}

/// One node as application 0 of a one-node block (IR_SPEC 11.2, 11.3).
fn run_node(p: &Json) -> Result<Reply, String> {
    let node = json::strip_prose(p.get("node").ok_or("no node")?);
    let o = op(&node).to_string();
    if o == "depth_route" {
        return Ok(Reply::Skip("depth_route has no single-node form".into()));
    }
    let mut weights = HashMap::new();
    if let Some(Json::Obj(ws)) = p.get("weights") {
        for (k, v) in ws {
            weights.insert(k.clone(), tensor_from_json(v)?);
        }
    }
    let mut env: ops::Env = HashMap::new();
    let mut rows_n: Option<usize> = None;
    if let Some(Json::Obj(is)) = p.get("inputs") {
        for (k, v) in is {
            let val = match v {
                Json::Arr(a) if a.iter().all(|x| matches!(x, Json::Num(_))) => Val::I(a.iter().map(|x| x.as_i64().unwrap_or(0)).collect()),
                Json::Arr(a) => Val::L(a.iter().map(tensor_from_json).collect::<Result<_, _>>()?),
                t => {
                    let mut t = tensor_from_json(t)?;
                    if t.shape.len() == 1 {
                        t.shape = vec![1, t.shape[0]];
                    }
                    Val::T(t)
                }
            };
            env.insert(k.clone(), val);
        }
    }
    for n in model::ins(&node) {
        match env.get(&n) {
            Some(Val::T(t)) => {
                rows_n.get_or_insert(t.rows());
            }
            Some(Val::I(v)) if n == "tokens" => {
                rows_n.get_or_insert(v.len());
            }
            Some(Val::L(l)) if !l.is_empty() => {
                rows_n.get_or_insert(l[0].rows());
            }
            _ => {}
        }
    }
    let t_n = rows_n.unwrap_or(0);
    let pos: Vec<usize> = match env.get("pos") {
        Some(Val::I(v)) => v.iter().map(|&x| x.max(0) as usize).collect(),
        _ => (0..t_n).collect(),
    };
    // the slot: declared by the request, or the op's own at its initial value
    let mut slots_decl: Vec<Json> = Vec::new();
    let mut pre: Slots = Slots::default();
    let sname = node.get("state").and_then(|s| s.as_str()).map(String::from);
    let given = p.get("state").filter(|s| !s.is_null());
    if let Some(sn) = &sname {
        let decl = given.and_then(|g| g.get(sn)).and_then(|s| s.get("decl")).cloned();
        let decl = match decl {
            Some(d) => d,
            None => default_slot(&node, sn)?,
        };
        slots_decl.push(decl);
    }
    let mut model = Model {
        vocab: 0,
        d: 0,
        ctx: 0,
        repeat: 1,
        embed: vec![],
        block: vec![node.clone()],
        head: vec![],
        slots: slots_decl.clone(),
        weight_index: vec![0],
        weights,
        state_k: 0,
        has_state_weight: node.get("state_weight").is_some(),
        recs: HashMap::new(),
        single_node: true,
    };
    model.compile_recs()?;
    if let (Some(sn), Some(g)) = (&sname, given) {
        if let Some(Json::Obj(ts)) = g.get(sn).and_then(|s| s.get("tensors")) {
            let names = state_names(&model, &node, 1, 0);
            let mut v = vec![Tensor::zeros(vec![0]); names.len()];
            for (k, t) in ts {
                if let Some(i) = names.iter().position(|n| n == k) {
                    v[i] = tensor_from_json(t)?;
                }
            }
            pre.rec.insert((sn.clone(), 0), v);
        }
    }
    let rows = ops::Rows { pos };
    let ctx = ops::Ctx { model: &model, app: Some(0), rows: &rows, codes: None, section: 1, node_index: 0 };
    ops::exec(&ctx, &node, &mut env, &mut pre)?;
    let mut outj = Vec::new();
    for n in outs(&node) {
        match env.get(&n) {
            Some(Val::T(t)) => outj.push((n, tensor_to_json(t))),
            Some(Val::L(l)) => outj.push((n, Json::Arr(l.iter().map(tensor_to_json).collect()))),
            _ => return Err(format!("the node wrote no {}", n)),
        }
    }
    let mut statej = Vec::new();
    if let Some(sn) = &sname {
        if let Some(ts) = pre.rec.get(&(sn.clone(), 0)) {
            let names = state_names(&model, &node, 1, 0);
            let decl = slot_tensors(&slots_decl[0]);
            let mut tobj = Vec::new();
            for (k, t) in names.iter().zip(ts) {
                let shape = decl.iter().find(|(dn, _)| dn == k).map(|x| x.1.clone()).unwrap_or(t.shape.clone());
                tobj.push((k.clone(), tensor_to_json(&Tensor::new(shape, t.data.clone()))));
            }
            statej.push((sn.clone(), Json::Obj(tobj)));
        }
    }
    Ok(Reply::Result(Json::Obj(vec![("outputs".into(), Json::Obj(outj)), ("state".into(), Json::Obj(statej))])))
}

/// The slot a single node needs when the request gives none.
fn default_slot(node: &Json, name: &str) -> Result<Json, String> {
    let gi = |k: &str| node.get(k).and_then(|v| v.as_i64()).unwrap_or(0);
    let t = |n: &str, shape: Vec<i64>| Json::obj(vec![("name", Json::str(n)), ("shape", Json::Arr(shape.into_iter().map(Json::int).collect())), ("init", Json::int(0))]);
    let rec = |op_: &str, tensors: Vec<Json>| {
        Json::obj(vec![("name", Json::str(name)), ("kind", Json::str("recurrent")), ("op", Json::str(op_)), ("dtype", Json::str("f32")), ("tensors", Json::Arr(tensors))])
    };
    Ok(match op(node) {
        "attention" => Json::obj(vec![
            ("name", Json::str(name)),
            ("kind", Json::str("kv")),
            ("heads", Json::int(gi("heads"))),
            ("kv_heads", Json::int(gi("kv_heads"))),
            ("head_dim", Json::int(gi("head_dim"))),
            ("max_len", Json::int(1 << 30)),
            ("window", Json::int(gi("window"))),
            ("sinks", Json::int(gi("sinks"))),
        ]),
        "linear_attention" => rec("linear_attention", vec![t("num", vec![gi("heads"), gi("head_dim")]), t("den", vec![gi("heads"), gi("head_dim")])]),
        "ssm_scan" => rec("ssm_scan", vec![t("s", vec![gi("n")])]),
        "conv1d_causal" => rec("conv1d_causal", vec![t("x", vec![gi("kernel") - 1, gi("channels")])]),
        "selective_scan" => rec("selective_scan", vec![t("s", vec![gi("channels"), gi("d_state")])]),
        "recurrence" => {
            let r = recurrence::Rec::compile(node)?;
            rec("recurrence", r.state_names().into_iter().map(|(n, d, _)| t(&n, d.into_iter().map(|x| x as i64).collect())).collect())
        }
        o => return Err(format!("{} has no slot", o)),
    })
}

fn reply_line(id: &Json, r: Reply) -> String {
    let body = match r {
        Reply::Result(v) => ("result", v),
        Reply::Error(e) => ("error", Json::Str(e)),
        Reply::Skip(s) => ("skip", Json::Str(s)),
    };
    json::to_string(&Json::Obj(vec![("id".into(), id.clone()), (body.0.into(), body.1)]))
}

fn conformance() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut eng = Engine { sessions: HashMap::new(), next: 0 };
    let log = std::env::var("KCR_LOG").ok().and_then(|p| std::fs::OpenOptions::new().create(true).append(true).open(p).ok());
    let mut log = log;
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let t0 = std::time::Instant::now();
        let (id, out) = match json::parse_value(&line) {
            Err(e) => (Json::Null, Reply::Error(format!("bad request: {}", e))),
            Ok(req) => {
                let id = req.get("id").cloned().unwrap_or(Json::Null);
                let method = p_str(&req, "method").unwrap_or("").to_string();
                let params = req.get("params").cloned().unwrap_or(Json::Obj(vec![]));
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| eng.handle(&method, &params)));
                let r = r.unwrap_or_else(|e| {
                    let m = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()));
                    Reply::Error(format!("internal error: {}", m.unwrap_or_default()))
                });
                if let Some(f) = log.as_mut() {
                    let short: String = line.chars().take(300).collect();
                    let tag = match &r {
                        Reply::Result(_) => "ok".to_string(),
                        Reply::Error(e) => format!("ERROR {}", e),
                        Reply::Skip(s) => format!("SKIP {}", s),
                    };
                    let _ = writeln!(f, "{} [{:?}] -> {}", short, t0.elapsed(), tag);
                }
                (id, r)
            }
        };
        let mut o = stdout.lock();
        let _ = writeln!(o, "{}", reply_line(&id, out));
        let _ = o.flush();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("--conformance") | None => conformance(),
        Some("check") if args.len() >= 3 => {
            let ir = args.get(3).map(|s| s.as_str());
            match load(&args[2], ir) {
                Ok((g, doc, warns)) => {
                    for w in warns {
                        eprintln!("warning: {}", w);
                    }
                    match build_model(&g, &doc) {
                        Ok(_) => println!("accepted"),
                        Err(e) => {
                            println!("refused: {}", e);
                            std::process::exit(1)
                        }
                    }
                }
                Err(e) => {
                    println!("refused: {}", e);
                    std::process::exit(1)
                }
            }
        }
        Some("validate") if args.len() >= 3 => {
            let text = std::fs::read_to_string(&args[2]).expect("read");
            let doc = json::parse_value(&text).expect("json");
            let rep = validate::validate(&doc, None);
            for e in &rep.errors {
                println!("{}", e);
            }
            if rep.errors.is_empty() {
                println!("accepted");
            } else {
                std::process::exit(1)
            }
        }
        Some("numbers") if args.len() >= 3 => {
            // one binary64 bit pattern (16 hex digits) per line -> its ECMAScript form
            let text = std::fs::read_to_string(&args[2]).expect("read");
            let mut out = String::new();
            for l in text.lines() {
                let x = f64::from_bits(u64::from_str_radix(l.trim(), 16).expect("hex"));
                out.push_str(&json::es_number(x));
                out.push('\n');
            }
            print!("{}", out);
        }
        Some("canon") if args.len() >= 3 => match digest::digest(&args[2], true) {
            Ok((d, c)) => println!("{} {}", c, d),
            Err(e) => println!("refused: {}", e),
        },
        Some("digest") if args.len() >= 3 => {
            let text = std::fs::read_to_string(&args[2]).expect("read");
            match digest::digest(&text, false) {
                Ok((d, _)) => println!("{}", d),
                Err(e) => {
                    println!("refused: {}", e);
                    std::process::exit(1)
                }
            }
        }
        _ => {
            eprintln!("usage: kcr --conformance | kcr check FILE.gguf [DOC.ir.json] | kcr validate DOC.ir.json | kcr digest DOC");
            std::process::exit(2)
        }
    }
}
