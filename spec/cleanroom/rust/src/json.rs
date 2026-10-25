//! A JSON parser and serialiser written for this implementation (no crates).
//!
//! The parser keeps member order and duplicate members, remembers whether a number was written in
//! integer form, and records every departure from I-JSON (RFC 7493) it meets, so the digest (IR_SPEC
//! 3.10) can refuse a text without a second parser.

use std::fmt::Write as _;

#[derive(Clone, Debug)]
pub struct Number {
    pub v: f64,
    /// Written without a fraction or an exponent.
    pub int_form: bool,
}

#[derive(Clone, Debug)]
pub enum Json {
    Null,
    Bool(bool),
    Num(Number),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            // a duplicate member name: the last one holds (as ECMAScript's JSON.parse and Python's
            // json do; IR_SPEC does not say, FINDINGS.md)
            Json::Obj(m) => m.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(n.v),
            _ => None,
        }
    }
    /// An integer value (any number with no fractional part).
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Num(n) if n.v.is_finite() && n.v.fract() == 0.0 && n.v.abs() < 9.3e18 => Some(n.v as i64),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&Vec<Json>> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_obj(&self) -> Option<&Vec<(String, Json)>> {
        match self {
            Json::Obj(o) => Some(o),
            _ => None,
        }
    }
    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }
    pub fn num(v: f64) -> Json {
        Json::Num(Number { v, int_form: v.fract() == 0.0 })
    }
    pub fn int(v: i64) -> Json {
        Json::Num(Number { v: v as f64, int_form: true })
    }
    pub fn str(s: &str) -> Json {
        Json::Str(s.to_string())
    }
    pub fn obj(items: Vec<(&str, Json)>) -> Json {
        Json::Obj(items.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }
    pub fn type_name(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "boolean",
            Json::Num(_) => "number",
            Json::Str(_) => "string",
            Json::Arr(_) => "array",
            Json::Obj(_) => "object",
        }
    }
}

/// Equality as JSON values (numbers by value, objects as unordered maps).
pub fn json_eq(a: &Json, b: &Json) -> bool {
    match (a, b) {
        (Json::Null, Json::Null) => true,
        (Json::Bool(x), Json::Bool(y)) => x == y,
        (Json::Num(x), Json::Num(y)) => x.v == y.v,
        (Json::Str(x), Json::Str(y)) => x == y,
        (Json::Arr(x), Json::Arr(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| json_eq(p, q)),
        (Json::Obj(x), Json::Obj(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.iter().any(|(k2, v2)| k == k2 && json_eq(v, v2)))
        }
        _ => false,
    }
}

pub struct Parsed {
    pub value: Json,
    /// I-JSON departures (RFC 7493 and IR_SPEC 3.10 step 1), with where they are.
    pub ijson_issues: Vec<String>,
}

struct P<'a> {
    s: &'a [u8],
    i: usize,
    issues: Vec<String>,
    depth: usize,
}

pub fn parse(text: &str) -> Result<Parsed, String> {
    let mut p = P { s: text.as_bytes(), i: 0, issues: Vec::new(), depth: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("trailing characters at byte {}", p.i));
    }
    Ok(Parsed { value: v, ijson_issues: p.issues })
}

/// Parses and refuses any I-JSON departure only where `strict`; duplicate members are an error either way
/// for documents (a document is an object with closed keys).
pub fn parse_value(text: &str) -> Result<Json, String> {
    parse(text).map(|p| p.value)
}

/// A document's text (IR_SPEC 3.2): a duplicate member name anywhere, `_` members included, is refused
/// (rule 35); the other I-JSON departures matter only to the digest.
pub fn parse_document(text: &str) -> Result<Json, String> {
    let p = parse(text).map_err(|e| format!("IR_SPEC 3.2 (rule 1): the document is not JSON: {}", e))?;
    if let Some(d) = p.ijson_issues.iter().find(|m| m.starts_with("duplicate member name")) {
        return Err(format!("IR_SPEC 3.2 (rule 35): {}", d));
    }
    Ok(p.value)
}

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn err<T>(&self, m: &str) -> Result<T, String> {
        Err(format!("{} at byte {}", m, self.i))
    }
    fn value(&mut self) -> Result<Json, String> {
        if self.i >= self.s.len() {
            return self.err("unexpected end");
        }
        match self.s[self.i] {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Json::Str(self.string()?)),
            b't' => self.lit("true", Json::Bool(true)),
            b'f' => self.lit("false", Json::Bool(false)),
            b'n' => self.lit("null", Json::Null),
            b'-' | b'0'..=b'9' => self.number(),
            _ => self.err("not JSON"),
        }
    }
    fn lit(&mut self, w: &str, v: Json) -> Result<Json, String> {
        if self.s[self.i..].starts_with(w.as_bytes()) {
            self.i += w.len();
            Ok(v)
        } else {
            self.err("not JSON")
        }
    }
    fn object(&mut self) -> Result<Json, String> {
        self.depth += 1;
        if self.depth > 10000 {
            return self.err("nested too deep");
        }
        self.i += 1;
        let mut m: Vec<(String, Json)> = Vec::new();
        self.ws();
        if self.i < self.s.len() && self.s[self.i] == b'}' {
            self.i += 1;
            self.depth -= 1;
            return Ok(Json::Obj(m));
        }
        loop {
            self.ws();
            if self.i >= self.s.len() || self.s[self.i] != b'"' {
                return self.err("expected a member name");
            }
            let k = self.string()?;
            self.ws();
            if self.i >= self.s.len() || self.s[self.i] != b':' {
                return self.err("expected ':'");
            }
            self.i += 1;
            self.ws();
            let v = self.value()?;
            if m.iter().any(|(k2, _)| *k2 == k) {
                self.issues.push(format!("duplicate member name {:?}", k));
            }
            m.push((k, v));
            self.ws();
            if self.i >= self.s.len() {
                return self.err("unexpected end in object");
            }
            match self.s[self.i] {
                b',' => self.i += 1,
                b'}' => {
                    self.i += 1;
                    break;
                }
                _ => return self.err("expected ',' or '}'"),
            }
        }
        self.depth -= 1;
        Ok(Json::Obj(m))
    }
    fn array(&mut self) -> Result<Json, String> {
        self.depth += 1;
        if self.depth > 10000 {
            return self.err("nested too deep");
        }
        self.i += 1;
        let mut a = Vec::new();
        self.ws();
        if self.i < self.s.len() && self.s[self.i] == b']' {
            self.i += 1;
            self.depth -= 1;
            return Ok(Json::Arr(a));
        }
        loop {
            self.ws();
            a.push(self.value()?);
            self.ws();
            if self.i >= self.s.len() {
                return self.err("unexpected end in array");
            }
            match self.s[self.i] {
                b',' => self.i += 1,
                b']' => {
                    self.i += 1;
                    break;
                }
                _ => return self.err("expected ',' or ']'"),
            }
        }
        self.depth -= 1;
        Ok(Json::Arr(a))
    }
    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.s.len() {
            return self.err("bad \\u escape");
        }
        let h = std::str::from_utf8(&self.s[self.i..self.i + 4]).map_err(|_| "bad \\u escape".to_string())?;
        let v = u32::from_str_radix(h, 16).map_err(|_| format!("bad \\u escape at byte {}", self.i))?;
        self.i += 4;
        Ok(v)
    }
    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // opening quote
        let mut out = String::new();
        loop {
            if self.i >= self.s.len() {
                return self.err("unterminated string");
            }
            let c = self.s[self.i];
            match c {
                b'"' => {
                    self.i += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.i += 1;
                    if self.i >= self.s.len() {
                        return self.err("bad escape");
                    }
                    let e = self.s[self.i];
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let u = self.hex4()?;
                            if (0xD800..0xDC00).contains(&u) {
                                // a high surrogate: needs a low one next
                                if self.s[self.i..].starts_with(b"\\u") {
                                    let save = self.i;
                                    self.i += 2;
                                    let lo = self.hex4()?;
                                    if (0xDC00..0xE000).contains(&lo) {
                                        let cp = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
                                        out.push(char::from_u32(cp).unwrap());
                                        continue;
                                    }
                                    self.i = save;
                                }
                                self.issues.push("a lone surrogate (not well-formed Unicode)".into());
                                out.push('\u{FFFD}');
                            } else if (0xDC00..0xE000).contains(&u) {
                                self.issues.push("a lone surrogate (not well-formed Unicode)".into());
                                out.push('\u{FFFD}');
                            } else {
                                out.push(char::from_u32(u).unwrap());
                            }
                        }
                        _ => return self.err("bad escape"),
                    }
                }
                0x00..=0x1f => return self.err("control character in string"),
                _ => {
                    // copy one UTF-8 sequence (the input is a Rust str, so it is well formed)
                    let start = self.i;
                    let len = if c < 0x80 {
                        1
                    } else if c >> 5 == 0b110 {
                        2
                    } else if c >> 4 == 0b1110 {
                        3
                    } else {
                        4
                    };
                    self.i += len;
                    out.push_str(std::str::from_utf8(&self.s[start..self.i]).unwrap());
                }
            }
        }
    }
    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        let mut neg = false;
        if self.s[self.i] == b'-' {
            neg = true;
            self.i += 1;
        }
        let ds = self.i;
        if self.i < self.s.len() && self.s[self.i] == b'0' {
            self.i += 1;
        } else if self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
            while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
                self.i += 1;
            }
        } else {
            return self.err("not JSON (bad number)");
        }
        let int_digits = std::str::from_utf8(&self.s[ds..self.i]).unwrap().to_string();
        let mut int_form = true;
        if self.i < self.s.len() && self.s[self.i] == b'.' {
            int_form = false;
            self.i += 1;
            let f = self.i;
            while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
                self.i += 1;
            }
            if self.i == f {
                return self.err("bad fraction");
            }
        }
        if self.i < self.s.len() && (self.s[self.i] == b'e' || self.s[self.i] == b'E') {
            int_form = false;
            self.i += 1;
            if self.i < self.s.len() && (self.s[self.i] == b'+' || self.s[self.i] == b'-') {
                self.i += 1;
            }
            let f = self.i;
            while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
                self.i += 1;
            }
            if self.i == f {
                return self.err("bad exponent");
            }
        }
        let lex = std::str::from_utf8(&self.s[start..self.i]).unwrap();
        let v: f64 = lex.parse().map_err(|_| format!("bad number {}", lex))?;
        if !v.is_finite() {
            self.issues.push(format!("a number beyond the double range ({})", lex));
        }
        if int_form {
            // I-JSON: integers within -2^53 .. 2^53 (inclusive), judged on the exact digits
            let d = int_digits.trim_start_matches('0');
            let limit = "9007199254740992";
            if d.len() > limit.len() || (d.len() == limit.len() && d > limit) {
                self.issues.push(format!("an integer beyond 2^53 ({})", lex));
            }
        } else if v == 0.0 && neg {
            self.issues.push(format!("a negative zero in floating form ({})", lex));
        }
        Ok(Json::Num(Number { v, int_form }))
    }
}

// ---------------------------------------------------------------- serialisation

/// ECMAScript Number.prototype.toString of a finite binary64 value (RFC 8785 section 3.2.2.3).
pub fn es_number(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    if x < 0.0 {
        return format!("-{}", es_number(-x));
    }
    // Rust's `{:e}` prints the shortest round-trip digits: d.ddd e N
    let e = format!("{:e}", x);
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let mut digits: String = mant.chars().filter(|c| *c != '.').collect();
    let k = digits.len() as i32;
    let n = exp + 1;
    // ECMAScript: of two equally close shortest digit strings, the even one. Rust breaks the tie
    // upward (1424953923781206.25 prints as ...206.3; ECMAScript writes ...206.2).
    if let Some(alt) = even_tie(x, &digits, n) {
        digits = alt;
    }
    if k <= n && n <= 21 {
        let mut s = digits.clone();
        for _ in 0..(n - k) {
            s.push('0');
        }
        s
    } else if 0 < n && n <= 21 {
        format!("{}.{}", &digits[..n as usize], &digits[n as usize..])
    } else if -6 < n && n <= 0 {
        let mut s = String::from("0.");
        for _ in 0..(-n) {
            s.push('0');
        }
        s.push_str(&digits);
        s
    } else {
        let sign = if n - 1 >= 0 { '+' } else { '-' };
        let ee = (n - 1).abs();
        if k == 1 {
            format!("{}e{}{}", digits, sign, ee)
        } else {
            format!("{}.{}e{}{}", &digits[..1], &digits[1..], sign, ee)
        }
    }
}

/// The exact decimal digits of a positive finite binary64 value: (digits, n) with x = 0.digits x 10^n,
/// no leading or trailing zeros.
fn exact_decimal(x: f64) -> (String, i32) {
    let bits = x.to_bits();
    let exp_bits = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    let (m, e) = if exp_bits == 0 { (frac, -1074) } else { (frac | (1u64 << 52), exp_bits - 1075) };
    // big integer in base 1e9, little-endian
    let mut big: Vec<u64> = vec![m % 1_000_000_000, (m / 1_000_000_000) % 1_000_000_000, m / 1_000_000_000_000_000_000];
    let mul = |big: &mut Vec<u64>, f: u64| {
        let mut carry = 0u64;
        for d in big.iter_mut() {
            let v = *d * f + carry;
            *d = v % 1_000_000_000;
            carry = v / 1_000_000_000;
        }
        while carry > 0 {
            big.push(carry % 1_000_000_000);
            carry /= 1_000_000_000;
        }
    };
    let mut point_shift = 0i32; // value = big * 10^(-point_shift)
    if e >= 0 {
        for _ in 0..e {
            mul(&mut big, 2);
        }
    } else {
        for _ in 0..(-e) {
            mul(&mut big, 5);
        }
        point_shift = -e;
    }
    while big.len() > 1 && *big.last().unwrap() == 0 {
        big.pop();
    }
    let mut s = format!("{}", big.last().unwrap());
    for d in big.iter().rev().skip(1) {
        s.push_str(&format!("{:09}", d));
    }
    let n = s.len() as i32 - point_shift;
    let t = s.trim_end_matches('0').to_string();
    (t, n)
}

fn even_tie(x: f64, digits: &str, n: i32) -> Option<String> {
    let last = *digits.as_bytes().last()?;
    if (last - b'0') % 2 == 0 {
        return None;
    }
    let (ex, en) = exact_decimal(x);
    if en != n || ex.len() <= digits.len() {
        return None;
    }
    let k = digits.len();
    let (head, rest) = ex.split_at(k);
    if rest != "5" {
        return None;
    }
    // a tie between head and head + 1: take the even one
    let candidate = if head == digits {
        // Rust kept the lower one; it is odd: the upper is even
        let mut b = head.as_bytes().to_vec();
        let mut i = k;
        loop {
            if i == 0 {
                return None;
            }
            i -= 1;
            if b[i] == b'9' {
                b[i] = b'0';
            } else {
                b[i] += 1;
                break;
            }
        }
        String::from_utf8(b).ok()?
    } else {
        head.to_string()
    };
    // it must read back as x
    let text = format!("0.{}e{}", candidate, n);
    if text.parse::<f64>().ok()? == x && candidate.as_bytes()[k - 1] % 2 == 0 {
        Some(candidate)
    } else {
        None
    }
}

/// A string as ECMAScript's JSON.stringify writes it.
pub fn es_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn utf16_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

/// RFC 8785 (JCS) serialisation.
pub fn canonical(v: &Json, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(n) => out.push_str(&es_number(n.v)),
        Json::Str(s) => es_string(s, out),
        Json::Arr(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                canonical(x, out);
            }
            out.push(']');
        }
        Json::Obj(m) => {
            let mut idx: Vec<usize> = (0..m.len()).collect();
            idx.sort_by(|&i, &j| utf16_cmp(&m[i].0, &m[j].0));
            out.push('{');
            for (n, &i) in idx.iter().enumerate() {
                if n > 0 {
                    out.push(',');
                }
                es_string(&m[i].0, out);
                out.push(':');
                canonical(&m[i].1, out);
            }
            out.push('}');
        }
    }
}

/// Compact serialisation in member order (for replies).
pub fn write(v: &Json, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(n) => {
            if n.v.is_finite() {
                if n.int_form && n.v.abs() < 1e15 {
                    let _ = write!(out, "{}", n.v as i64);
                } else {
                    out.push_str(&es_number(n.v));
                }
            } else {
                out.push_str("null");
            }
        }
        Json::Str(s) => es_string(s, out),
        Json::Arr(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write(x, out);
            }
            out.push(']');
        }
        Json::Obj(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                es_string(k, out);
                out.push(':');
                write(x, out);
            }
            out.push('}');
        }
    }
}

pub fn to_string(v: &Json) -> String {
    let mut s = String::new();
    write(v, &mut s);
    s
}

/// Removes every member whose name starts with '_', at any depth (IR_SPEC 3.2, 3.10 step 2).
pub fn strip_prose(v: &Json) -> Json {
    match v {
        Json::Obj(m) => Json::Obj(
            m.iter().filter(|(k, _)| !k.starts_with('_')).map(|(k, x)| (k.clone(), strip_prose(x))).collect(),
        ),
        Json::Arr(a) => Json::Arr(a.iter().map(strip_prose).collect()),
        x => x.clone(),
    }
}
