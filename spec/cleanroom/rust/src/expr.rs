//! The `expr` grammar (IR_SPEC 5.19) and its extension for recurrence bodies (IR_SPEC 6.10.2).
//!
//! One parser serves both: in expression mode names are positional inputs (x, a, b, ..., v); in body
//! mode names are left unresolved (tensor reads `name[...]`, bare index names, `sum(j, e)`) for the
//! recurrence compiler to resolve.

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fun {
    Exp,
    Log,
    Sqrt,
    Tanh,
    Erf,
    Abs,
    Sigmoid,
    Relu,
    Max,
    Min,
    Where,
    RowSum,
    RowMean,
    RowMax,
}

impl Fun {
    pub fn from_name(s: &str) -> Option<Fun> {
        Some(match s {
            "exp" => Fun::Exp,
            "log" => Fun::Log,
            "sqrt" => Fun::Sqrt,
            "tanh" => Fun::Tanh,
            "erf" => Fun::Erf,
            "abs" => Fun::Abs,
            "sigmoid" => Fun::Sigmoid,
            "relu" => Fun::Relu,
            "max" => Fun::Max,
            "min" => Fun::Min,
            "where" => Fun::Where,
            "rowsum" => Fun::RowSum,
            "rowmean" => Fun::RowMean,
            "rowmax" => Fun::RowMax,
            _ => return None,
        })
    }
    pub fn arity(self) -> usize {
        match self {
            Fun::Max | Fun::Min => 2,
            Fun::Where => 3,
            _ => 1,
        }
    }
    pub fn is_row(self) -> bool {
        matches!(self, Fun::RowSum | Fun::RowMean | Fun::RowMax)
    }
    /// The libm transcendentals that make a body ROUNDING (IR_SPEC 6.10.4; `^` too).
    pub fn is_transcendental(self) -> bool {
        matches!(self, Fun::Exp | Fun::Log | Fun::Tanh | Fun::Erf | Fun::Sigmoid)
    }
}

pub const FUNCTION_NAMES: [&str; 14] =
    ["exp", "log", "sqrt", "tanh", "erf", "abs", "sigmoid", "relu", "max", "min", "where", "rowsum", "rowmean", "rowmax"];

#[derive(Clone, Debug)]
pub enum RawIx {
    /// index name plus an integer offset
    Name(String, i64),
    Const(i64),
}

#[derive(Clone, Debug)]
pub enum Ex {
    Num(f32),
    /// expression mode: the k-th input (x = 0, a = 1, ...)
    Input(usize),
    /// body mode: a bare name (must be an index in scope)
    Name(String, usize),
    /// body mode: a tensor read
    Read(String, Vec<RawIx>, usize),
    /// body mode: sum over an index
    Sum(String, Box<Ex>, usize),
    Neg(Box<Ex>),
    Bin(u8, Box<Ex>, Box<Ex>),
    Call(Fun, Vec<Ex>),
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f32),
    Id(String),
    Int(i64, bool), // an integer literal (also kept as a number), whether it was written as digits only
    Sym(u8),
    End,
}

/// IR_SPEC 5.19: every engine accepts nesting up to 200 deep; the measure of depth is not defined,
/// so this parser counts parentheses, calls, unary minus and exponents, and accepts far more.
pub const MAX_DEPTH: usize = 1000;

struct Lexer {
    toks: Vec<(Tok, usize)>,
}

fn lex(src: &str) -> Result<Lexer, String> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut toks = Vec::new();
    while i < b.len() {
        let c = b[i];
        if c == b' ' || c == b'\t' {
            i += 1;
            continue;
        }
        let col = i + 1;
        if c.is_ascii_digit() || (c == b'.' && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
            let s = i;
            let mut digits_only = true;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            if i < b.len() && b[i] == b'.' {
                digits_only = false;
                i += 1;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                let mut j = i + 1;
                if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
                    j += 1;
                }
                if j < b.len() && b[j].is_ascii_digit() {
                    digits_only = false;
                    i = j;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let lexeme = &src[s..i];
            // strtof in the C locale: normalise "2." and ".5" for Rust's parser (correctly rounded)
            let mut norm = lexeme.to_string();
            if norm.starts_with('.') {
                norm.insert(0, '0');
            }
            if let Some(p) = norm.find('.') {
                let after = norm.as_bytes().get(p + 1).copied();
                if after.is_none() || !after.unwrap().is_ascii_digit() {
                    norm.insert(p + 1, '0');
                }
            }
            let v: f32 = norm.parse().map_err(|_| format!("column {}: bad number {:?}", col, lexeme))?;
            if !v.is_finite() {
                return Err(format!("column {}: number {:?} is not finite in f32", col, lexeme));
            }
            if digits_only {
                let iv: i64 = lexeme.parse().unwrap_or(i64::MAX);
                toks.push((Tok::Int(iv, true), col));
                // keep the float value too: stored by the parser from the lexeme
                toks.last_mut().unwrap().0 = Tok::Int(iv, true);
                // the f32 value of an integer literal is computed again from the lexeme when used
                let _ = v;
            } else {
                toks.push((Tok::Num(v), col));
            }
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            let s = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            toks.push((Tok::Id(src[s..i].to_string()), col));
            continue;
        }
        if b"+-*/^(),[]=".contains(&c) {
            toks.push((Tok::Sym(c), col));
            i += 1;
            continue;
        }
        return Err(format!("column {}: unexpected character {:?}", col, src[i..].chars().next().unwrap()));
    }
    toks.push((Tok::End, b.len() + 1));
    Ok(Lexer { toks })
}

fn int_value(iv: i64, src_col: usize, src: &str) -> f32 {
    // an integer literal read as the nearest float (strtof): parse its digits as f32
    let s = &src[src_col - 1..];
    let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    s[..end].parse::<f32>().unwrap_or(iv as f32)
}

pub struct Parser<'a> {
    src: &'a str,
    toks: Vec<(Tok, usize)>,
    pos: usize,
    body: bool,
    n_inputs: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str, body: bool, n_inputs: usize) -> Result<Parser<'a>, String> {
        let l = lex(src)?;
        Ok(Parser { src, toks: l.toks, pos: 0, body, n_inputs, depth: 0 })
    }
    fn peek(&self) -> &Tok {
        &self.toks[self.pos].0
    }
    fn col(&self) -> usize {
        self.toks[self.pos].1
    }
    fn next(&mut self) -> Tok {
        let t = self.toks[self.pos].0.clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }
    fn err<T>(&self, m: &str) -> Result<T, String> {
        Err(format!("column {}: {}", self.col(), m))
    }
    fn expect(&mut self, c: u8) -> Result<(), String> {
        if *self.peek() == Tok::Sym(c) {
            self.next();
            Ok(())
        } else {
            self.err(&format!("expected '{}'", c as char))
        }
    }
    fn enter(&mut self) -> Result<(), String> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return self.err(&format!("nested deeper than {}", MAX_DEPTH));
        }
        Ok(())
    }
    pub fn expr(&mut self) -> Result<Ex, String> {
        let mut a = self.term()?;
        loop {
            match self.peek() {
                Tok::Sym(b'+') | Tok::Sym(b'-') => {
                    let op = if let Tok::Sym(c) = self.next() { c } else { unreachable!() };
                    let b = self.term()?;
                    a = Ex::Bin(op, Box::new(a), Box::new(b));
                }
                _ => break,
            }
        }
        Ok(a)
    }
    fn term(&mut self) -> Result<Ex, String> {
        let mut a = self.unary()?;
        loop {
            match self.peek() {
                Tok::Sym(b'*') | Tok::Sym(b'/') => {
                    let op = if let Tok::Sym(c) = self.next() { c } else { unreachable!() };
                    let b = self.unary()?;
                    a = Ex::Bin(op, Box::new(a), Box::new(b));
                }
                _ => break,
            }
        }
        Ok(a)
    }
    fn unary(&mut self) -> Result<Ex, String> {
        if *self.peek() == Tok::Sym(b'-') {
            self.next();
            self.enter()?;
            let u = self.unary()?;
            self.depth -= 1;
            return Ok(Ex::Neg(Box::new(u)));
        }
        self.power()
    }
    fn power(&mut self) -> Result<Ex, String> {
        let p = self.primary()?;
        if *self.peek() == Tok::Sym(b'^') {
            self.next();
            self.enter()?;
            let u = self.unary()?;
            self.depth -= 1;
            return Ok(Ex::Bin(b'^', Box::new(p), Box::new(u)));
        }
        Ok(p)
    }
    fn ix(&mut self) -> Result<RawIx, String> {
        match self.next() {
            Tok::Int(v, true) => Ok(RawIx::Const(v)),
            Tok::Id(n) => {
                let mut off = 0i64;
                if let Tok::Sym(c @ (b'+' | b'-')) = *self.peek() {
                    self.next();
                    match self.next() {
                        Tok::Int(v, true) => off = if c == b'+' { v } else { -v },
                        _ => {
                            self.pos -= 1;
                            return self.err("an index offset must be an integer");
                        }
                    }
                }
                Ok(RawIx::Name(n, off))
            }
            _ => {
                self.pos -= 1;
                self.err("expected an index")
            }
        }
    }
    fn primary(&mut self) -> Result<Ex, String> {
        let col = self.col();
        match self.next() {
            Tok::Num(v) => Ok(Ex::Num(v)),
            Tok::Int(iv, _) => Ok(Ex::Num(int_value(iv, col, self.src))),
            Tok::Sym(b'(') => {
                self.enter()?;
                let e = self.expr()?;
                self.depth -= 1;
                self.expect(b')')?;
                Ok(e)
            }
            Tok::Id(name) => {
                if name == "pi" {
                    return Ok(Ex::Num(std::f32::consts::PI));
                }
                if self.body && name == "sum" && *self.peek() == Tok::Sym(b'(') {
                    self.next();
                    self.enter()?;
                    let idx = match self.next() {
                        Tok::Id(n) => n,
                        _ => {
                            self.pos -= 1;
                            return self.err("sum needs an index name");
                        }
                    };
                    self.expect(b',')?;
                    let e = self.expr()?;
                    self.expect(b')')?;
                    self.depth -= 1;
                    return Ok(Ex::Sum(idx, Box::new(e), col));
                }
                if let Some(f) = Fun::from_name(&name) {
                    if *self.peek() != Tok::Sym(b'(') {
                        self.pos -= 1;
                        return self.err(&format!("function {} needs arguments", name));
                    }
                    if self.body && f.is_row() {
                        self.pos -= 1;
                        return self.err(&format!("{} is not part of the recurrence grammar (sum replaces it)", name));
                    }
                    self.next();
                    self.enter()?;
                    let mut args = vec![self.expr()?];
                    while *self.peek() == Tok::Sym(b',') {
                        self.next();
                        args.push(self.expr()?);
                    }
                    self.expect(b')')?;
                    self.depth -= 1;
                    if args.len() != f.arity() {
                        return Err(format!(
                            "column {}: {} takes {} argument(s), not {}",
                            col,
                            name,
                            f.arity(),
                            args.len()
                        ));
                    }
                    return Ok(Ex::Call(f, args));
                }
                if self.body {
                    if *self.peek() == Tok::Sym(b'[') {
                        self.next();
                        let mut ixs = vec![self.ix()?];
                        while *self.peek() == Tok::Sym(b',') {
                            self.next();
                            ixs.push(self.ix()?);
                        }
                        self.expect(b']')?;
                        return Ok(Ex::Read(name, ixs, col));
                    }
                    return Ok(Ex::Name(name, col));
                }
                // expression mode: x, a, b, ..., v
                let k = if name == "x" {
                    Some(0)
                } else if name.len() == 1 && (b'a'..=b'v').contains(&name.as_bytes()[0]) {
                    Some((name.as_bytes()[0] - b'a') as usize + 1)
                } else {
                    None
                };
                match k {
                    Some(k) if k < self.n_inputs => Ok(Ex::Input(k)),
                    Some(_) => Err(format!("column {}: name {} has no input (the node reads {})", col, name, self.n_inputs)),
                    None => Err(format!("column {}: unknown name {}", col, name)),
                }
            }
            Tok::End => Err(format!("column {}: unexpected end of expression", col)),
            t => Err(format!("column {}: unexpected {:?}", col, t)),
        }
    }
    fn finish(&self) -> Result<(), String> {
        if *self.peek() != Tok::End {
            return self.err("unexpected text after the expression");
        }
        Ok(())
    }
}

/// Parses an `expr` attribute over `n_inputs` inputs.
pub fn parse_expr(src: &str, n_inputs: usize) -> Result<Ex, String> {
    let mut p = Parser::new(src, false, n_inputs)?;
    let e = p.expr()?;
    p.finish()?;
    Ok(e)
}

/// A recurrence body.
pub fn parse_body(src: &str) -> Result<Ex, String> {
    let mut p = Parser::new(src, true, 0)?;
    let e = p.expr()?;
    p.finish()?;
    Ok(e)
}

/// `name[i1, ..., ik]` (a declaration) or `name[i1, ...] = body` (an assignment).
pub fn parse_decl(src: &str) -> Result<(String, Vec<String>), String> {
    let mut p = Parser::new(src, true, 0)?;
    let (n, ix) = decl_head(&mut p)?;
    p.finish()?;
    Ok((n, ix))
}

fn decl_head(p: &mut Parser) -> Result<(String, Vec<String>), String> {
    let name = match p.next() {
        Tok::Id(n) => n,
        _ => {
            p.pos = p.pos.saturating_sub(1);
            return p.err("expected a tensor name");
        }
    };
    p.expect(b'[')?;
    let mut ix = Vec::new();
    loop {
        match p.next() {
            Tok::Id(n) => ix.push(n),
            _ => {
                p.pos -= 1;
                return p.err("expected an index name");
            }
        }
        match p.next() {
            Tok::Sym(b',') => continue,
            Tok::Sym(b']') => break,
            _ => {
                p.pos -= 1;
                return p.err("expected ',' or ']'");
            }
        }
    }
    Ok((name, ix))
}

pub fn parse_assignment(src: &str) -> Result<(String, Vec<String>, Ex), String> {
    let mut p = Parser::new(src, true, 0)?;
    let (n, ix) = decl_head(&mut p)?;
    p.expect(b'=')?;
    let e = p.expr()?;
    p.finish()?;
    Ok((n, ix, e))
}

// ---------------------------------------------------------------- numerics

/// erf in f64, accurate to a few ulps of f64 (series below 2.5, continued fraction above).
pub fn erf64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let ax = x.abs();
    let r = if ax < 2.5 {
        let x2 = ax * ax;
        let mut term = ax; // x^(2n+1) / n! with sign
        let mut sum = 0.0;
        let mut n = 0u32;
        loop {
            let t = term / (2 * n + 1) as f64;
            sum += t;
            if t.abs() <= 1e-18 * sum.abs() || n > 300 {
                break;
            }
            n += 1;
            term *= -x2 / n as f64;
        }
        sum * (2.0 / std::f64::consts::PI.sqrt())
    } else if ax < 7.0 {
        // erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + ...))))
        let mut f = ax;
        for k in (1..=80).rev() {
            f = ax + (k as f64 / 2.0) / f;
        }
        1.0 - (-ax * ax).exp() / std::f64::consts::PI.sqrt() / f
    } else {
        1.0
    };
    if x < 0.0 {
        -r
    } else {
        r
    }
}

pub fn erf32(x: f32) -> f32 {
    erf64(x as f64) as f32
}

pub fn sigmoid(x: f32) -> f32 {
    let e = (-x).exp();
    let d = 1.0f32 + e;
    1.0f32 / d
}

pub fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

pub fn fmax(a: f32, b: f32) -> f32 {
    if a >= b {
        a
    } else {
        b
    }
}

pub fn fmin(a: f32, b: f32) -> f32 {
    if a <= b {
        a
    } else {
        b
    }
}

pub fn apply1(f: Fun, x: f32) -> f32 {
    match f {
        Fun::Exp => x.exp(),
        Fun::Log => x.ln(),
        Fun::Sqrt => x.sqrt(),
        Fun::Tanh => x.tanh(),
        Fun::Erf => erf32(x),
        Fun::Abs => x.abs(),
        Fun::Sigmoid => sigmoid(x),
        Fun::Relu => relu(x),
        _ => unreachable!(),
    }
}

pub fn binop(op: u8, a: f32, b: f32) -> f32 {
    match op {
        b'+' => a + b,
        b'-' => a - b,
        b'*' => a * b,
        b'/' => a / b,
        b'^' => a.powf(b),
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------- the expr op over [T, w] values

#[derive(Clone)]
enum V {
    S(f32),
    M(Vec<f32>),
}

fn map1(v: V, f: impl Fn(f32) -> f32) -> V {
    match v {
        V::S(s) => V::S(f(s)),
        V::M(m) => V::M(m.into_iter().map(f).collect()),
    }
}

fn map2(a: V, b: V, f: impl Fn(f32, f32) -> f32) -> V {
    match (a, b) {
        (V::S(x), V::S(y)) => V::S(f(x, y)),
        (V::S(x), V::M(y)) => V::M(y.into_iter().map(|q| f(x, q)).collect()),
        (V::M(x), V::S(y)) => V::M(x.into_iter().map(|p| f(p, y)).collect()),
        (V::M(x), V::M(y)) => V::M(x.into_iter().zip(y).map(|(p, q)| f(p, q)).collect()),
    }
}

fn get(v: &V, i: usize) -> f32 {
    match v {
        V::S(s) => *s,
        V::M(m) => m[i],
    }
}

fn eval_v(e: &Ex, ins: &[&[f32]], t: usize, w: usize) -> V {
    match e {
        Ex::Num(v) => V::S(*v),
        Ex::Input(k) => V::M(ins[*k].to_vec()),
        Ex::Neg(a) => map1(eval_v(a, ins, t, w), |x| -x),
        Ex::Bin(op, a, b) => {
            let op = *op;
            map2(eval_v(a, ins, t, w), eval_v(b, ins, t, w), move |x, y| binop(op, x, y))
        }
        Ex::Call(f, args) => match f {
            Fun::Max => map2(eval_v(&args[0], ins, t, w), eval_v(&args[1], ins, t, w), fmax),
            Fun::Min => map2(eval_v(&args[0], ins, t, w), eval_v(&args[1], ins, t, w), fmin),
            Fun::Where => {
                let c = eval_v(&args[0], ins, t, w);
                let a = eval_v(&args[1], ins, t, w);
                let b = eval_v(&args[2], ins, t, w);
                if let (V::S(c), V::S(a), V::S(b)) = (&c, &a, &b) {
                    return V::S(if *c > 0.0 { *a } else { *b });
                }
                V::M((0..t * w).map(|i| if get(&c, i) > 0.0 { get(&a, i) } else { get(&b, i) }).collect())
            }
            Fun::RowSum | Fun::RowMean | Fun::RowMax => {
                let a = eval_v(&args[0], ins, t, w);
                let reduce = |row: &[f32]| -> f32 {
                    match f {
                        Fun::RowMax => {
                            let mut m = row[0];
                            for &x in &row[1..] {
                                if x > m {
                                    m = x;
                                }
                            }
                            m
                        }
                        _ => {
                            let mut s = 0.0f32;
                            for &x in row {
                                s += x;
                            }
                            if *f == Fun::RowMean {
                                s / w as f32
                            } else {
                                s
                            }
                        }
                    }
                };
                match a {
                    V::S(s) => V::S(reduce(&vec![s; w])),
                    V::M(m) => {
                        let mut out = Vec::with_capacity(t * w);
                        for r in 0..t {
                            let v = reduce(&m[r * w..(r + 1) * w]);
                            out.extend(std::iter::repeat_n(v, w));
                        }
                        V::M(out)
                    }
                }
            }
            f => map1(eval_v(&args[0], ins, t, w), |x| apply1(*f, x)),
        },
        _ => unreachable!("body node in an expr"),
    }
}

/// Evaluates an `expr` over inputs [T, w] (row-major), returning [T, w].
pub fn eval_expr(e: &Ex, ins: &[&[f32]], t: usize, w: usize) -> Vec<f32> {
    match eval_v(e, ins, t, w) {
        V::S(s) => vec![s; t * w],
        V::M(m) => m,
    }
}

/// Function names an expression uses (for capability and class decisions).
pub fn uses(e: &Ex, out: &mut Vec<Fun>, pow: &mut bool) {
    match e {
        Ex::Neg(a) => uses(a, out, pow),
        Ex::Bin(op, a, b) => {
            if *op == b'^' {
                *pow = true;
            }
            uses(a, out, pow);
            uses(b, out, pow);
        }
        Ex::Call(f, args) => {
            out.push(*f);
            for a in args {
                uses(a, out, pow);
            }
        }
        Ex::Sum(_, a, _) => uses(a, out, pow),
        _ => {}
    }
}
