//! The `recurrence` op (IR_SPEC 6.10): compile a node's declarations and bodies, check rules 31-33,
//! and run the sequential fold.

use crate::expr::{self, Ex, Fun, RawIx};
use crate::json::Json;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Input,
    Weight,
    State,
    Let,
    Out,
}

#[derive(Clone, Debug)]
pub struct TDecl {
    pub name: String,
    pub kind: Kind,
    pub dims: Vec<usize>,
}

#[derive(Clone, Debug)]
enum CIx {
    Var(usize, i64),
    Const(i64),
}

#[derive(Clone, Debug)]
enum CEx {
    Num(f32),
    Idx(usize),
    Read(usize, Vec<CIx>),
    Sum(usize, Box<CEx>),
    Neg(Box<CEx>),
    Bin(u8, Box<CEx>, Box<CEx>),
    Call(Fun, Vec<CEx>),
}

#[derive(Clone, Debug)]
struct Assign {
    tensor: usize,
    idx: Vec<usize>,
    body: CEx,
}

#[derive(Clone, Debug)]
pub struct Rec {
    pub idx_names: Vec<String>,
    pub idx_size: Vec<usize>,
    pub tensors: Vec<TDecl>,
    /// tensor ids of the inputs, in the order of `in`
    pub inputs: Vec<usize>,
    /// (tensor id, tensor name in the file, `{i}` not yet replaced)
    pub weights: Vec<(usize, String)>,
    init: Vec<Assign>,
    lets: Vec<Assign>,
    updates: Vec<Assign>,
    readout: Assign,
    /// state tensor ids, in update order
    pub state: Vec<usize>,
    pub exact: bool,
}

const RESERVED: [&str; 4] = ["pi", "sum", "per_layer", "rowsum"];

fn is_ident(s: &str) -> bool {
    let b = s.as_bytes();
    !b.is_empty() && (b[0].is_ascii_alphabetic() || b[0] == b'_') && b.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'_')
}

fn rerr<T>(rule: u32, m: String) -> Result<T, String> {
    Err(format!("IR_SPEC 6.10 (rule {}): {}", rule, m))
}

#[derive(Clone, Copy, PartialEq)]
enum Section {
    Init,
    Let(usize),
    Update,
    Readout,
}

struct Ctx<'a> {
    rec: &'a Rec,
    section: Section,
    scope: Vec<usize>,
    what: String,
}

impl<'a> Ctx<'a> {
    fn idx_id(&self, n: &str) -> Option<usize> {
        self.rec.idx_names.iter().position(|x| x == n)
    }
    fn tensor_id(&self, n: &str) -> Option<usize> {
        self.rec.tensors.iter().position(|t| t.name == n)
    }
    fn can_read(&self, t: usize) -> bool {
        let k = self.rec.tensors[t].kind;
        match self.section {
            Section::Init => k == Kind::Weight,
            Section::Let(order) => match k {
                Kind::Input | Kind::Weight | Kind::State => true,
                Kind::Let => self.rec.lets.iter().take(order).any(|a| a.tensor == t) || self.let_before(t, order),
                Kind::Out => false,
            },
            Section::Update => matches!(k, Kind::Input | Kind::Weight | Kind::State | Kind::Let),
            Section::Readout => matches!(k, Kind::Input | Kind::Weight | Kind::Let | Kind::State),
        }
    }
    fn let_before(&self, t: usize, order: usize) -> bool {
        // lets are registered as tensors in order; `order` is this let's position
        let lets: Vec<usize> = self.rec.tensors.iter().enumerate().filter(|(_, d)| d.kind == Kind::Let).map(|(i, _)| i).collect();
        lets.iter().position(|&x| x == t).map(|p| p < order).unwrap_or(false)
    }
    fn compile(&mut self, e: &Ex) -> Result<CEx, String> {
        Ok(match e {
            Ex::Num(v) => CEx::Num(*v),
            Ex::Input(_) => unreachable!(),
            Ex::Name(n, col) => {
                let id = self.idx_id(n);
                match id {
                    Some(i) if self.scope.contains(&i) => CEx::Idx(i),
                    Some(_) => return rerr(32, format!("{}: column {}: index {} is not in scope", self.what, col, n)),
                    None => {
                        return rerr(
                            32,
                            format!("{}: column {}: bare name {} is not an index in scope (a tensor is read with its indices)", self.what, col, n),
                        )
                    }
                }
            }
            Ex::Read(n, ixs, col) => {
                let t = match self.tensor_id(n) {
                    Some(t) => t,
                    None => return rerr(32, format!("{}: column {}: {} is not a declared tensor", self.what, col, n)),
                };
                if !self.can_read(t) {
                    return rerr(32, format!("{}: column {}: this body cannot read {}", self.what, col, n));
                }
                let rank = self.rec.tensors[t].dims.len();
                if ixs.len() != rank {
                    return rerr(32, format!("{}: column {}: {} has {} dimension(s), read with {} index(es)", self.what, col, n, rank, ixs.len()));
                }
                let mut c = Vec::new();
                for ix in ixs {
                    c.push(match ix {
                        RawIx::Const(v) => CIx::Const(*v),
                        RawIx::Name(m, off) => match self.idx_id(m) {
                            Some(i) if self.scope.contains(&i) => CIx::Var(i, *off),
                            Some(_) => return rerr(32, format!("{}: column {}: index {} is not in scope", self.what, col, m)),
                            None => return rerr(32, format!("{}: column {}: index {} is not declared", self.what, col, m)),
                        },
                    });
                }
                CEx::Read(t, c)
            }
            Ex::Sum(j, a, col) => {
                let i = match self.idx_id(j) {
                    Some(i) => i,
                    None => return rerr(32, format!("{}: column {}: sum over undeclared index {}", self.what, col, j)),
                };
                if self.scope.contains(&i) {
                    return rerr(32, format!("{}: column {}: sum over {}, already in scope", self.what, col, j));
                }
                self.scope.push(i);
                let b = self.compile(a)?;
                self.scope.pop();
                CEx::Sum(i, Box::new(b))
            }
            Ex::Neg(a) => CEx::Neg(Box::new(self.compile(a)?)),
            Ex::Bin(op, a, b) => CEx::Bin(*op, Box::new(self.compile(a)?), Box::new(self.compile(b)?)),
            Ex::Call(f, args) => {
                if f.is_row() {
                    return rerr(31, format!("{}: {:?} is not part of the recurrence grammar", self.what, f));
                }
                CEx::Call(*f, args.iter().map(|a| self.compile(a)).collect::<Result<_, _>>()?)
            }
        })
    }
}

fn str_list(node: &Json, key: &str) -> Result<Vec<String>, String> {
    match node.get(key) {
        None => Ok(Vec::new()),
        Some(Json::Arr(a)) => a.iter().map(|x| x.as_str().map(String::from).ok_or(format!("{} holds a non-string", key))).collect(),
        Some(_) => Err(format!("{} is not a list of strings", key)),
    }
}

impl Rec {
    /// Compiles a recurrence node (rules 31-33 that need no slot or file).
    pub fn compile(node: &Json) -> Result<Rec, String> {
        let n_in = node.get("in").and_then(|x| x.as_arr()).map(|a| a.len()).unwrap_or(0);
        let index = match node.get("index") {
            Some(Json::Obj(m)) => m.clone(),
            _ => return rerr(31, "index is not a map of names to positive integers".into()),
        };
        let mut rec = Rec {
            idx_names: Vec::new(),
            idx_size: Vec::new(),
            tensors: Vec::new(),
            inputs: Vec::new(),
            weights: Vec::new(),
            init: Vec::new(),
            lets: Vec::new(),
            updates: Vec::new(),
            readout: Assign { tensor: 0, idx: vec![], body: CEx::Num(0.0) },
            state: Vec::new(),
            exact: true,
        };
        let mut all_names: Vec<String> = Vec::new();
        let declare = |n: &str, all: &mut Vec<String>| -> Result<(), String> {
            if !is_ident(n) {
                return rerr(31, format!("{:?} is not an identifier", n));
            }
            if RESERVED.contains(&n) || expr::FUNCTION_NAMES.contains(&n) {
                return rerr(31, format!("{} is a reserved name", n));
            }
            if all.iter().any(|x| x == n) {
                return rerr(31, format!("{} is declared twice", n));
            }
            all.push(n.to_string());
            Ok(())
        };
        for (k, v) in &index {
            if k.starts_with('_') {
                continue;
            }
            declare(k, &mut all_names)?;
            let s = match v.as_i64() {
                Some(s) if s >= 1 => s as usize,
                _ => return rerr(31, format!("index {} has size {}, not a positive integer", k, crate::json::to_string(v))),
            };
            rec.idx_names.push(k.clone());
            rec.idx_size.push(s);
        }
        let size_of = |rec: &Rec, ix: &[String], what: &str| -> Result<Vec<usize>, String> {
            ix.iter()
                .map(|n| match rec.idx_names.iter().position(|x| x == n) {
                    Some(i) => Ok(rec.idx_size[i]),
                    None => rerr(32, format!("{}: index {} is not declared", what, n)),
                })
                .collect()
        };
        // inputs
        let inputs = str_list(node, "inputs").map_err(|e| format!("IR_SPEC 6.10 (rule 31): {}", e))?;
        if inputs.is_empty() || inputs.len() != n_in {
            return rerr(31, format!("inputs declares {} value(s), in has {}", inputs.len(), n_in));
        }
        for d in &inputs {
            let (n, ix) = expr::parse_decl(d).map_err(|e| format!("IR_SPEC 6.10 (rule 31): input {:?}: {}", d, e))?;
            declare(&n, &mut all_names)?;
            let dims = size_of(&rec, &ix, d)?;
            rec.inputs.push(rec.tensors.len());
            rec.tensors.push(TDecl { name: n, kind: Kind::Input, dims });
        }
        // weights
        match node.get("weights") {
            None => {}
            Some(Json::Obj(m)) => {
                for (d, tn) in m {
                    if d.starts_with('_') {
                        continue;
                    }
                    let tn = tn.as_str().ok_or_else(|| format!("IR_SPEC 6.10 (rule 31): weight {} does not name a tensor", d))?;
                    let (n, ix) = expr::parse_decl(d).map_err(|e| format!("IR_SPEC 6.10 (rule 31): weight {:?}: {}", d, e))?;
                    declare(&n, &mut all_names)?;
                    let dims = size_of(&rec, &ix, d)?;
                    rec.weights.push((rec.tensors.len(), tn.to_string()));
                    rec.tensors.push(TDecl { name: n, kind: Kind::Weight, dims });
                }
            }
            Some(_) => return rerr(31, "weights is not a map of declarations to tensor names".into()),
        }
        // assignments: parse all first
        let parse_assigns = |key: &str| -> Result<Vec<(String, Vec<String>, Ex, String)>, String> {
            let list = str_list(node, key).map_err(|e| format!("IR_SPEC 6.10 (rule 31): {}", e))?;
            list.iter()
                .map(|s| {
                    expr::parse_assignment(s)
                        .map(|(n, ix, e)| (n, ix, e, s.clone()))
                        .map_err(|e| format!("IR_SPEC 6.10 (rule 31): {} body {:?} does not parse: {}", key, s, e))
                })
                .collect()
        };
        let init = parse_assigns("init")?;
        let lets = parse_assigns("let")?;
        let updates = parse_assigns("update")?;
        if updates.is_empty() {
            return rerr(31, "update is missing or empty".into());
        }
        let readout_src = match node.get("readout") {
            Some(Json::Str(s)) => s.clone(),
            _ => return rerr(31, "readout is not one assignment".into()),
        };
        let readout = expr::parse_assignment(&readout_src)
            .map_err(|e| format!("IR_SPEC 6.10 (rule 31): readout {:?} does not parse: {}", readout_src, e))?;

        let check_assign_idx = |rec: &Rec, ix: &[String], what: &str| -> Result<Vec<usize>, String> {
            let mut ids = Vec::new();
            for n in ix {
                match rec.idx_names.iter().position(|x| x == n) {
                    Some(i) => {
                        if ids.contains(&i) {
                            return rerr(32, format!("{}: the assignment repeats index {}", what, n));
                        }
                        ids.push(i)
                    }
                    None => return rerr(32, format!("{}: index {} is not declared", what, n)),
                }
            }
            Ok(ids)
        };
        // state tensors (from update), lets, readout output: declare
        for (n, ix, _, src) in &updates {
            declare(n, &mut all_names)?;
            let ids = check_assign_idx(&rec, ix, src)?;
            let dims = ids.iter().map(|&i| rec.idx_size[i]).collect();
            rec.state.push(rec.tensors.len());
            rec.tensors.push(TDecl { name: n.clone(), kind: Kind::State, dims });
        }
        for (n, ix, _, src) in &lets {
            declare(n, &mut all_names)?;
            let ids = check_assign_idx(&rec, ix, src)?;
            let dims = ids.iter().map(|&i| rec.idx_size[i]).collect();
            rec.tensors.push(TDecl { name: n.clone(), kind: Kind::Let, dims });
        }
        {
            let (n, ix, _) = &readout;
            declare(n, &mut all_names)?;
            let ids = check_assign_idx(&rec, ix, &readout_src)?;
            let dims = ids.iter().map(|&i| rec.idx_size[i]).collect();
            rec.tensors.push(TDecl { name: n.clone(), kind: Kind::Out, dims });
        }
        // compile bodies
        let compile = |rec: &Rec, sec: Section, n: &str, ix: &[String], e: &Ex, src: &str| -> Result<Assign, String> {
            let t = rec.tensors.iter().position(|d| d.name == n).unwrap();
            let ids = check_assign_idx(rec, ix, src)?;
            let mut ctx = Ctx { rec, section: sec, scope: ids.clone(), what: format!("{:?}", src) };
            let body = ctx.compile(e)?;
            Ok(Assign { tensor: t, idx: ids, body })
        };
        let mut inits = Vec::new();
        let mut seen_init: Vec<String> = Vec::new();
        for (n, ix, e, src) in &init {
            let t = rec.tensors.iter().position(|d| d.name == *n && d.kind == Kind::State);
            let t = match t {
                Some(t) => t,
                None => return rerr(33, format!("init {:?} assigns {}, which is not a state tensor", src, n)),
            };
            if seen_init.contains(n) {
                return rerr(33, format!("init assigns {} twice", n));
            }
            seen_init.push(n.clone());
            let ids = check_assign_idx(&rec, ix, src)?;
            let dims: Vec<usize> = ids.iter().map(|&i| rec.idx_size[i]).collect();
            if dims != rec.tensors[t].dims {
                return rerr(33, format!("init {:?} assigns {} with other sizes than its update", src, n));
            }
            inits.push(compile(&rec, Section::Init, n, ix, e, src)?);
        }
        let mut lets_c = Vec::new();
        for (k, (n, ix, e, src)) in lets.iter().enumerate() {
            lets_c.push(compile(&rec, Section::Let(k), n, ix, e, src)?);
            rec.lets = lets_c.clone();
        }
        let mut ups = Vec::new();
        for (n, ix, e, src) in &updates {
            ups.push(compile(&rec, Section::Update, n, ix, e, src)?);
        }
        let ro = compile(&rec, Section::Readout, &readout.0, &readout.1, &readout.2, &readout_src)?;
        rec.init = inits;
        rec.lets = lets_c;
        rec.updates = ups;
        rec.readout = ro;
        // class (6.10.4)
        let mut funs = Vec::new();
        let mut pow = false;
        for (_, _, e, _) in init.iter().chain(lets.iter()).chain(updates.iter()) {
            expr::uses(e, &mut funs, &mut pow);
        }
        expr::uses(&readout.2, &mut funs, &mut pow);
        rec.exact = !pow && !funs.iter().any(|f| f.is_transcendental());
        Ok(rec)
    }

    pub fn out_width(&self) -> usize {
        self.tensors[self.readout.tensor].dims.iter().product()
    }
    pub fn input_width(&self, k: usize) -> usize {
        self.tensors[self.inputs[k]].dims.iter().product()
    }
    pub fn state_names(&self) -> Vec<(String, Vec<usize>, bool)> {
        self.state
            .iter()
            .map(|&t| (self.tensors[t].name.clone(), self.tensors[t].dims.clone(), self.init.iter().any(|a| a.tensor == t)))
            .collect()
    }

    /// The initial state: each tensor's init body, or +0.
    pub fn initial_state(&self, weights: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut store: Vec<Vec<f32>> = self.tensors.iter().map(|d| vec![0.0; d.dims.iter().product()]).collect();
        for (k, (t, _)) in self.weights.iter().enumerate() {
            store[*t] = weights[k].clone();
        }
        let mut out = Vec::new();
        for &s in &self.state {
            if let Some(a) = self.init.iter().find(|a| a.tensor == s) {
                out.push(self.eval_assign(a, &store, &store));
            } else {
                out.push(vec![0.0; self.tensors[s].dims.iter().product()]);
            }
        }
        out
    }

    /// Runs T tokens. `ins[k]` is input k's [T, w] row-major data; `state` is updated in place.
    pub fn run(&self, ins: &[&[f32]], t_rows: usize, weights: &[Vec<f32>], state: &mut [Vec<f32>]) -> Vec<f32> {
        let mut store: Vec<Vec<f32>> = self.tensors.iter().map(|d| vec![0.0; d.dims.iter().product()]).collect();
        for (k, (t, _)) in self.weights.iter().enumerate() {
            store[*t] = weights[k].clone();
        }
        let w_out = self.out_width();
        let mut out = Vec::with_capacity(t_rows * w_out);
        for t in 0..t_rows {
            for (k, &tid) in self.inputs.iter().enumerate() {
                let w = self.input_width(k);
                store[tid] = ins[k][t * w..(t + 1) * w].to_vec();
            }
            for (k, &sid) in self.state.iter().enumerate() {
                store[sid] = state[k].clone();
            }
            for a in &self.lets {
                let v = self.eval_assign(a, &store, &store);
                store[a.tensor] = v;
            }
            let mut new_state = Vec::new();
            for a in &self.updates {
                new_state.push(self.eval_assign(a, &store, &store));
            }
            // the readout reads the state after the update
            let mut after = store.clone();
            for (k, &sid) in self.state.iter().enumerate() {
                after[sid] = new_state[k].clone();
            }
            let y = self.eval_assign(&self.readout, &after, &after);
            out.extend_from_slice(&y);
            for (k, v) in new_state.into_iter().enumerate() {
                state[k] = v;
            }
        }
        out
    }

    fn eval_assign(&self, a: &Assign, store: &[Vec<f32>], _st: &[Vec<f32>]) -> Vec<f32> {
        let dims: Vec<usize> = a.idx.iter().map(|&i| self.idx_size[i]).collect();
        let n: usize = dims.iter().product();
        let mut vals = vec![0i64; self.idx_names.len()];
        let mut out = Vec::with_capacity(n);
        let mut coord = vec![0usize; dims.len()];
        for _ in 0..n {
            for (k, &i) in a.idx.iter().enumerate() {
                vals[i] = coord[k] as i64;
            }
            out.push(self.eval(&a.body, store, &mut vals));
            // advance row-major (last index fastest)
            for k in (0..dims.len()).rev() {
                coord[k] += 1;
                if coord[k] < dims[k] {
                    break;
                }
                coord[k] = 0;
            }
        }
        out
    }

    fn eval(&self, e: &CEx, store: &[Vec<f32>], vals: &mut Vec<i64>) -> f32 {
        match e {
            CEx::Num(v) => *v,
            CEx::Idx(i) => vals[*i] as f32,
            CEx::Read(t, ixs) => {
                let dims = &self.tensors[*t].dims;
                let mut flat = 0usize;
                for (k, ix) in ixs.iter().enumerate() {
                    let c = match ix {
                        CIx::Var(i, off) => vals[*i] + off,
                        CIx::Const(c) => *c,
                    };
                    if c < 0 || c as usize >= dims[k] {
                        return 0.0;
                    }
                    flat = flat * dims[k] + c as usize;
                }
                store[*t][flat]
            }
            CEx::Sum(j, b) => {
                let mut s = 0.0f32;
                let save = vals[*j];
                for v in 0..self.idx_size[*j] {
                    vals[*j] = v as i64;
                    s += self.eval(b, store, vals);
                }
                vals[*j] = save;
                s
            }
            CEx::Neg(a) => -self.eval(a, store, vals),
            CEx::Bin(op, a, b) => {
                let x = self.eval(a, store, vals);
                let y = self.eval(b, store, vals);
                expr::binop(*op, x, y)
            }
            CEx::Call(f, args) => match f {
                Fun::Max => {
                    let a = self.eval(&args[0], store, vals);
                    let b = self.eval(&args[1], store, vals);
                    expr::fmax(a, b)
                }
                Fun::Min => {
                    let a = self.eval(&args[0], store, vals);
                    let b = self.eval(&args[1], store, vals);
                    expr::fmin(a, b)
                }
                Fun::Where => {
                    let c = self.eval(&args[0], store, vals);
                    let a = self.eval(&args[1], store, vals);
                    let b = self.eval(&args[2], store, vals);
                    if c > 0.0 {
                        a
                    } else {
                        b
                    }
                }
                f => {
                    let a = self.eval(&args[0], store, vals);
                    expr::apply1(*f, a)
                }
            },
        }
    }
}
