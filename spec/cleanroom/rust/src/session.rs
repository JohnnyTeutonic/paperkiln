//! Sessions (IR_SPEC 11.3): one sequence on one loaded model, run incrementally, with eviction
//! (6.8.3), truncation (6.9), snapshots, and Mixture-of-Recursions routing (7.4, 7.5).

use crate::json::Json;
use crate::model::{at, at_i64, at_str, op, Model};
use crate::ops::{self, Codes, Ctx, Env, Rows, Slots, Val};
use crate::tensor::Tensor;
use std::collections::HashMap;

pub struct Session {
    pub ids: Vec<i64>,
    pub slots: Slots,
    pub evict: bool,
    pub snapshots: HashMap<String, (Vec<i64>, Slots)>,
}

pub struct CallOut {
    pub logits: Tensor,
    pub depth: Vec<usize>,
}

pub fn uses_turn_state(m: &Model) -> bool {
    m.block.iter().any(|n| op(n) == "attention" && (0..m.repeat).any(|a| at_str(n, "mask", a) == Some("turn_state")))
}

pub fn evict_declared(m: &Model) -> bool {
    m.slots.iter().any(|s| s.get("evict").and_then(|e| e.as_str()) == Some("turn_state"))
}

fn has_recurrent(m: &Model) -> bool {
    m.slots.iter().any(|s| s.get("kind").and_then(|k| k.as_str()) == Some("recurrent"))
}

impl Session {
    pub fn new(evict: bool) -> Session {
        Session { ids: Vec::new(), slots: Slots::default(), evict, snapshots: HashMap::new() }
    }

    pub fn forward(&mut self, m: &Model, ids: &[i64]) -> Result<CallOut, String> {
        let backup = (self.ids.clone(), self.slots.clone());
        match self.forward_inner(m, ids) {
            Ok(o) => Ok(o),
            Err(e) => {
                self.ids = backup.0;
                self.slots = backup.1;
                Err(e)
            }
        }
    }

    fn codes(&self, m: &Model, all: &[i64]) -> Option<Codes> {
        if m.state_k > 0 || m.has_state_weight {
            Some(Codes::compute(all, m.vocab, m.state_k))
        } else {
            None
        }
    }

    fn forward_inner(&mut self, m: &Model, ids: &[i64]) -> Result<CallOut, String> {
        let base = self.ids.len();
        let n = ids.len();
        for &id in ids {
            if id < 0 {
                return Err(format!("id {} is negative (IR_SPEC 6.8.1)", id));
            }
        }
        let mut all = self.ids.clone();
        all.extend_from_slice(ids);
        let codes = self.codes(m, &all);
        let rows = Rows { pos: (base..base + n).collect() };
        let mut env: Env = HashMap::new();
        env.insert("tokens".into(), Val::I(ids.to_vec()));
        env.insert("pos".into(), Val::I(rows.pos.iter().map(|&p| p as i64).collect()));
        for (ni, node) in m.embed.iter().enumerate() {
            let ctx = Ctx { model: m, app: None, rows: &rows, codes: codes.as_ref(), section: 0, node_index: ni };
            ops::exec(&ctx, node, &mut env, &mut self.slots)?;
        }
        let depth = run_block(m, &mut env, &rows, codes.as_ref(), &mut self.slots)?;
        for (ni, node) in m.head.iter().enumerate() {
            let ctx = Ctx { model: m, app: None, rows: &rows, codes: codes.as_ref(), section: 2, node_index: ni };
            ops::exec(&ctx, node, &mut env, &mut self.slots)?;
        }
        let logits = match env.remove("logits") {
            Some(Val::T(t)) => t,
            _ => return Err("the head wrote no logits".into()),
        };
        if logits.rows() != n || logits.cols() != m.vocab {
            return Err(format!("logits are {:?}, not [{}, {}]", logits.shape, n, m.vocab));
        }
        self.ids = all;
        if self.evict {
            if let Some(c) = &codes {
                self.evict_rows(m, c);
            }
        }
        Ok(CallOut { logits, depth })
    }

    /// Eager eviction (IR_SPEC 6.8.3): after m complete blocks, drop the text rows of every segment < m.
    fn evict_rows(&mut self, m: &Model, c: &Codes) {
        let mtot = c.total;
        for s in &m.slots {
            if s.get("evict").and_then(|e| e.as_str()) != Some("turn_state") {
                continue;
            }
            let name = s.get("name").and_then(|n| n.as_str()).unwrap_or("");
            for a in 0..m.repeat {
                if let Some(inst) = self.slots.kv.get_mut(&(name.to_string(), a)) {
                    let keep: Vec<bool> = inst.pos.iter().map(|&p| c.c[p] || c.s[p] >= mtot).collect();
                    retain(inst, &keep);
                }
            }
        }
    }

    pub fn truncate(&mut self, m: &Model, n: usize) -> Result<(), String> {
        let len = self.ids.len();
        if n > len {
            return Err(format!("truncate to {} of a session of length {}", n, len));
        }
        if n == len {
            return Ok(());
        }
        if has_recurrent(m) {
            // a recurrence cannot drop tokens: run again from the start (IR_SPEC 6.9)
            let keep = self.ids[..n].to_vec();
            self.ids.clear();
            self.slots = Slots::default();
            if n > 0 {
                self.forward(m, &keep)?;
            }
            return Ok(());
        }
        if self.evict {
            let c = Codes::compute(&self.ids[..n], m.vocab, m.state_k);
            let seg = c.total;
            let start = c.segment_start(seg);
            let held = (start..n).filter(|&p| !c.c[p]).all(|p| self.slots.kv.values().all(|inst| inst.pos.contains(&p)));
            if !held {
                // the cut segment's text rows were dropped: drop segment >= m and re-prefill it
                for inst in self.slots.kv.values_mut() {
                    let keep: Vec<bool> = inst.pos.iter().map(|&p| p < start).collect();
                    retain(inst, &keep);
                }
                let replay = self.ids[start..n].to_vec();
                self.ids.truncate(start);
                if !replay.is_empty() {
                    self.forward(m, &replay)?;
                }
                return Ok(());
            }
        }
        for inst in self.slots.kv.values_mut() {
            let keep: Vec<bool> = inst.pos.iter().map(|&p| p < n).collect();
            retain(inst, &keep);
        }
        self.ids.truncate(n);
        Ok(())
    }
}

fn retain(inst: &mut ops::KvInst, keep: &[bool]) {
    let mut k = 0;
    inst.pos.retain(|_| {
        k += 1;
        keep[k - 1]
    });
    let mut k = 0;
    inst.k.retain(|_| {
        k += 1;
        keep[k - 1]
    });
    let mut k = 0;
    inst.v.retain(|_| {
        k += 1;
        keep[k - 1]
    });
}

fn gather_env(env: &Env, idx: &[usize], full: usize) -> Env {
    let mut sub = HashMap::new();
    for (k, v) in env {
        let nv = match v {
            Val::T(t) if t.rows() == full && t.shape.len() == 2 => Val::T(t.gather_rows(idx)),
            Val::L(l) => Val::L(l.iter().map(|t| if t.rows() == full { t.gather_rows(idx) } else { t.clone() }).collect()),
            Val::I(v) if v.len() == full => Val::I(idx.iter().map(|&i| v[i]).collect()),
            x => x.clone(),
        };
        sub.insert(k.clone(), nv);
    }
    sub
}

fn scatter_env(env: &mut Env, sub: Env, idx: &[usize], full: usize) {
    for (k, v) in sub {
        if let Val::T(t) = v {
            if t.rows() != idx.len() || t.shape.len() != 2 {
                continue;
            }
            let w = t.cols();
            let mut dst = match env.get(&k) {
                Some(Val::T(f)) if f.rows() == full && f.cols() == w => f.clone(),
                _ => Tensor::zeros(vec![full, w]),
            };
            for (r, &ti) in idx.iter().enumerate() {
                dst.data[ti * w..(ti + 1) * w].copy_from_slice(t.row(r));
            }
            env.insert(k, Val::T(dst));
        }
    }
}

fn step_of(dr: Option<&Json>, a: usize) -> i64 {
    match dr {
        Some(n) => at_i64(n, "step", a).unwrap_or(-1),
        None => -1,
    }
}

/// Runs the block's applications; under depth_route, each recursion step on its active tokens.
fn run_block(m: &Model, env: &mut Env, rows: &Rows, codes: Option<&Codes>, slots: &mut Slots) -> Result<Vec<usize>, String> {
    let dr = m.block.first().filter(|n| op(n) == "depth_route");
    let full = rows.pos.len();
    let mut depth: Option<Vec<usize>> = None;
    let mut gate: Vec<f32> = Vec::new();
    let mut x_in: Option<Tensor> = None;
    let mut active: Vec<usize> = Vec::new();
    for a in 0..m.repeat {
        let s = step_of(dr, a);
        let start = if dr.is_some() { 1 } else { 0 };
        if s < 0 {
            for (ni, node) in m.block.iter().enumerate().skip(start) {
                let ctx = Ctx { model: m, app: Some(a), rows, codes, section: 1, node_index: ni };
                ops::exec(&ctx, node, env, slots)?;
            }
            continue;
        }
        let drn = dr.unwrap();
        let k = s as usize;
        let first = a == 0 || step_of(dr, a - 1) != s;
        let last = a + 1 == m.repeat || step_of(dr, a + 1) != s;
        if depth.is_none() {
            // (a) routing, at the first application of step 0
            let x = match env.get("x") {
                Some(Val::T(t)) => t,
                _ => return Err("depth_route: no x".into()),
            };
            let w = m.wa(drn, "weight", Some(a))?;
            let r = at_i64(drn, "n_rec", a)? as usize;
            let logits = ops::linear(x, w, None)?;
            if logits.cols() != r {
                return Err("depth_route: router rows are not n_rec".into());
            }
            let mut dv = Vec::new();
            let mut gv = Vec::new();
            for t in 0..full {
                let l = logits.row(t);
                let mut j = 0;
                for q in 1..r {
                    if l[q] > l[j] {
                        j = q;
                    }
                }
                let mx = l[j];
                let e: Vec<f32> = l.iter().map(|&v| (v - mx).exp()).collect();
                let mut z = 0.0f32;
                for &v in &e {
                    z += v;
                }
                dv.push(j + 1);
                gv.push(e[j] / z);
            }
            let out = drn.get("out").and_then(|o| o.as_str()).unwrap_or("rec_depth").to_string();
            env.insert(out, Val::T(Tensor::mat(full, 1, dv.iter().map(|&d| d as f32).collect())));
            depth = Some(dv);
            gate = gv;
        }
        let dv = depth.as_ref().unwrap();
        if first {
            x_in = match env.get("x") {
                Some(Val::T(t)) => Some(t.clone()),
                _ => return Err("depth_route: no x".into()),
            };
            active = (0..full).filter(|&t| dv[t] > k).collect();
        }
        if active.is_empty() {
            continue; // (d) a step with no active token is skipped
        }
        let mut sub = gather_env(env, &active, full);
        let sub_rows = Rows { pos: active.iter().map(|&t| rows.pos[t]).collect() };
        for (ni, node) in m.block.iter().enumerate().skip(1) {
            let ctx = Ctx { model: m, app: Some(a), rows: &sub_rows, codes, section: 1, node_index: ni };
            ops::exec(&ctx, node, &mut sub, slots)?;
        }
        scatter_env(env, sub, &active, full);
        if last && at(drn, "gate", a).and_then(|g| g.as_str()) == Some("score") {
            let xi = x_in.as_ref().unwrap();
            if let Some(Val::T(x)) = env.get_mut("x") {
                let w = x.cols();
                for &t in &active {
                    for j in 0..w {
                        let xin = xi.data[t * w + j];
                        let diff = x.data[t * w + j] - xin;
                        let prod = gate[t] * diff;
                        x.data[t * w + j] = xin + prod;
                    }
                }
            }
        }
    }
    Ok(depth.unwrap_or_default())
}
