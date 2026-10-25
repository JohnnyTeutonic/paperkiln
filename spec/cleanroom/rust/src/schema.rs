//! A JSON Schema (2020-12) evaluator for the keywords `ir.schema.json` uses: type, enum, const,
//! required, properties, patternProperties (prefix patterns "^..."), additionalProperties, items,
//! minItems, minLength, minimum, $ref (local), anyOf, allOf, if/then, boolean schemas.

use crate::json::{json_eq, Json};

pub struct Schema {
    root: Json,
}

impl Schema {
    pub fn load() -> Schema {
        let text = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../ir.schema.json"));
        Schema { root: crate::json::parse_value(text).expect("ir.schema.json parses") }
    }

    fn resolve<'a>(&'a self, r: &str) -> Option<&'a Json> {
        let path = r.strip_prefix("#/")?;
        let mut cur = &self.root;
        for part in path.split('/') {
            cur = cur.get(part)?;
        }
        Some(cur)
    }

    /// Every violation, as "path: message".
    pub fn validate(&self, v: &Json) -> Vec<String> {
        let mut errs = Vec::new();
        self.check(&self.root, v, "$", &mut errs);
        errs
    }

    fn valid(&self, s: &Json, v: &Json) -> bool {
        let mut e = Vec::new();
        self.check(s, v, "", &mut e);
        e.is_empty()
    }

    fn check(&self, s: &Json, v: &Json, path: &str, errs: &mut Vec<String>) {
        match s {
            Json::Bool(true) => return,
            Json::Bool(false) => {
                errs.push(format!("{}: not allowed", path));
                return;
            }
            Json::Obj(_) => {}
            _ => return,
        }
        if let Some(Json::Str(r)) = s.get("$ref") {
            match self.resolve(r) {
                Some(t) => self.check(t, v, path, errs),
                None => errs.push(format!("{}: unresolved $ref {}", path, r)),
            }
        }
        if let Some(t) = s.get("type") {
            let ok = match t {
                Json::Str(n) => type_ok(n, v),
                Json::Arr(a) => a.iter().any(|x| x.as_str().map(|n| type_ok(n, v)).unwrap_or(false)),
                _ => true,
            };
            if !ok {
                errs.push(format!("{}: {} is not of type {}", path, short(v), crate::json::to_string(t)));
                return;
            }
        }
        if let Some(Json::Arr(e)) = s.get("enum") {
            if !e.iter().any(|x| json_eq(x, v)) {
                errs.push(format!("{}: {} is not one of {}", path, short(v), crate::json::to_string(&Json::Arr(e.clone()))));
            }
        }
        if let Some(c) = s.get("const") {
            if !json_eq(c, v) {
                errs.push(format!("{}: {} is not {}", path, short(v), crate::json::to_string(c)));
            }
        }
        if let (Some(m), Json::Num(n)) = (s.get("minimum").and_then(|x| x.as_f64()), v) {
            if n.v < m {
                errs.push(format!("{}: {} is below {}", path, n.v, m));
            }
        }
        if let (Some(m), Json::Str(st)) = (s.get("minLength").and_then(|x| x.as_i64()), v) {
            if (st.chars().count() as i64) < m {
                errs.push(format!("{}: string shorter than {}", path, m));
            }
        }
        if let Json::Arr(a) = v {
            if let Some(m) = s.get("minItems").and_then(|x| x.as_i64()) {
                if (a.len() as i64) < m {
                    errs.push(format!("{}: fewer than {} items", path, m));
                }
            }
            if let Some(it) = s.get("items") {
                for (i, x) in a.iter().enumerate() {
                    self.check(it, x, &format!("{}[{}]", path, i), errs);
                }
            }
        }
        if let Json::Obj(m) = v {
            if let Some(Json::Arr(req)) = s.get("required") {
                for r in req {
                    if let Some(r) = r.as_str() {
                        if !m.iter().any(|(k, _)| k == r) {
                            errs.push(format!("{}: required key {} is missing", path, r));
                        }
                    }
                }
            }
            let props = s.get("properties").and_then(|p| p.as_obj());
            let pats = s.get("patternProperties").and_then(|p| p.as_obj());
            for (k, x) in m {
                let mut matched = false;
                if let Some(p) = props {
                    if let Some((_, ps)) = p.iter().find(|(pk, _)| pk == k) {
                        matched = true;
                        self.check(ps, x, &format!("{}.{}", path, k), errs);
                    }
                }
                if let Some(p) = pats {
                    for (pat, ps) in p {
                        if pattern_matches(pat, k) {
                            matched = true;
                            self.check(ps, x, &format!("{}.{}", path, k), errs);
                        }
                    }
                }
                if !matched {
                    if let Some(ap) = s.get("additionalProperties") {
                        if let Json::Bool(false) = ap {
                            errs.push(format!("{}: key {} is not defined", path, k));
                        } else {
                            self.check(ap, x, &format!("{}.{}", path, k), errs);
                        }
                    }
                }
            }
        }
        if let Some(Json::Arr(all)) = s.get("allOf") {
            for sub in all {
                self.check(sub, v, path, errs);
            }
        }
        if let Some(Json::Arr(any)) = s.get("anyOf") {
            if !any.iter().any(|sub| self.valid(sub, v)) {
                // report the branch with the fewest errors, for a useful message
                let mut best: Option<Vec<String>> = None;
                for sub in any {
                    let mut e = Vec::new();
                    self.check(sub, v, path, &mut e);
                    if best.as_ref().map(|b| e.len() < b.len()).unwrap_or(true) {
                        best = Some(e);
                    }
                }
                errs.push(format!("{}: {} matches no allowed form ({})", path, short(v), best.unwrap_or_default().join("; ")));
            }
        }
        if let Some(cond) = s.get("if") {
            if self.valid(cond, v) {
                if let Some(t) = s.get("then") {
                    self.check(t, v, path, errs);
                }
            } else if let Some(e) = s.get("else") {
                self.check(e, v, path, errs);
            }
        }
    }
}

fn type_ok(n: &str, v: &Json) -> bool {
    match n {
        "object" => matches!(v, Json::Obj(_)),
        "array" => matches!(v, Json::Arr(_)),
        "string" => matches!(v, Json::Str(_)),
        "boolean" => matches!(v, Json::Bool(_)),
        "null" => matches!(v, Json::Null),
        "number" => matches!(v, Json::Num(_)),
        "integer" => matches!(v, Json::Num(x) if x.v.is_finite() && x.v.fract() == 0.0),
        _ => true,
    }
}

fn pattern_matches(pat: &str, k: &str) -> bool {
    // the schema uses only anchored literal prefixes ("^_")
    match pat.strip_prefix('^') {
        Some(lit) => k.starts_with(lit),
        None => k.contains(pat),
    }
}

fn short(v: &Json) -> String {
    let s = crate::json::to_string(v);
    if s.chars().count() > 80 {
        format!("{}...", s.chars().take(80).collect::<String>())
    } else {
        s
    }
}
