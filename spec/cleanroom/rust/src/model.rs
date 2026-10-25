//! A loaded document: its sections, slots, per-layer attributes and weights.

use crate::json::Json;
use crate::recurrence::Rec;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// An attribute at application `app` (a `per_layer` value reads entry app).
pub fn at<'a>(node: &'a Json, key: &str, app: usize) -> Option<&'a Json> {
    let v = node.get(key)?;
    if let Json::Obj(_) = v {
        if let Some(Json::Arr(a)) = v.get("per_layer") {
            return a.get(app);
        }
    }
    Some(v)
}

pub fn is_per_layer(v: &Json) -> bool {
    matches!(v, Json::Obj(_)) && v.get("per_layer").is_some()
}

pub fn at_f32(node: &Json, key: &str, app: usize) -> Result<f32, String> {
    at(node, key, app).and_then(|v| v.as_f64()).map(|v| v as f32).ok_or(format!("{}: attribute {} is not a number", op(node), key))
}

pub fn at_usize(node: &Json, key: &str, app: usize) -> Result<usize, String> {
    at(node, key, app)
        .and_then(|v| v.as_i64())
        .filter(|v| *v >= 0)
        .map(|v| v as usize)
        .ok_or(format!("{}: attribute {} is not a non-negative integer", op(node), key))
}

pub fn at_i64(node: &Json, key: &str, app: usize) -> Result<i64, String> {
    at(node, key, app).and_then(|v| v.as_i64()).ok_or(format!("{}: attribute {} is not an integer", op(node), key))
}

pub fn at_str<'a>(node: &'a Json, key: &str, app: usize) -> Option<&'a str> {
    at(node, key, app).and_then(|v| v.as_str())
}

pub fn op(node: &Json) -> &str {
    node.get("op").and_then(|v| v.as_str()).unwrap_or("?")
}

pub fn ins(node: &Json) -> Vec<String> {
    node.get("in")
        .and_then(|v| v.as_arr())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default()
}

pub fn outs(node: &Json) -> Vec<String> {
    match node.get("out") {
        Some(Json::Str(s)) => vec![s.clone()],
        Some(Json::Arr(a)) => a.iter().filter_map(|x| x.as_str().map(String::from)).collect(),
        _ => vec![],
    }
}

pub const STATEFUL: [&str; 6] = ["attention", "linear_attention", "ssm_scan", "conv1d_causal", "selective_scan", "recurrence"];

pub struct Model {
    pub vocab: usize,
    pub d: usize,
    pub ctx: usize,
    pub repeat: usize,
    pub embed: Vec<Json>,
    pub block: Vec<Json>,
    pub head: Vec<Json>,
    pub slots: Vec<Json>,
    /// m_i of each application (IR_SPEC 7.1; i when absent)
    pub weight_index: Vec<usize>,
    pub weights: HashMap<String, Tensor>,
    /// paperkiln.state_tokens.k (state ids vocab .. vocab + k - 1), 0 when absent
    pub state_k: usize,
    pub has_state_weight: bool,
    /// compiled recurrence nodes, by (section, node index)
    pub recs: HashMap<(u8, usize), Rec>,
    /// the run_node fixture mode: no model-level vocab and context
    pub single_node: bool,
}

impl Model {
    pub fn from_doc(doc: &Json, weights: HashMap<String, Tensor>, state_k: usize) -> Result<Model, String> {
        let m = doc.get("model").ok_or("no model")?;
        let num = |k: &str| m.get(k).and_then(|v| v.as_i64()).map(|v| v as usize).ok_or(format!("model.{} missing", k));
        let nodes = |sec: &str| -> Vec<Json> {
            doc.get(sec).and_then(|s| s.get("nodes")).and_then(|n| n.as_arr()).cloned().unwrap_or_default()
        };
        let block = doc.get("block").ok_or("no block")?;
        let repeat = block.get("repeat").and_then(|v| v.as_i64()).ok_or("no block.repeat")? as usize;
        let slots = block.get("state").and_then(|s| s.as_arr()).cloned().unwrap_or_default();
        let weight_index = match block.get("weight_index").and_then(|w| w.get("per_layer")).and_then(|p| p.as_arr()) {
            Some(a) => a.iter().map(|x| x.as_i64().unwrap_or(0) as usize).collect(),
            None => (0..repeat).collect(),
        };
        let embed = nodes("embed");
        let has_state_weight = embed.iter().any(|n| n.get("state_weight").is_some());
        let mut model = Model {
            vocab: num("vocab")?,
            d: num("d")?,
            ctx: num("ctx")?,
            repeat,
            embed,
            block: nodes("block"),
            head: nodes("head"),
            slots,
            weight_index,
            weights,
            state_k,
            has_state_weight,
            recs: HashMap::new(),
            single_node: false,
        };
        model.compile_recs()?;
        Ok(model)
    }

    pub fn compile_recs(&mut self) -> Result<(), String> {
        for (s, sec) in [&self.embed, &self.block, &self.head].iter().enumerate() {
            for (i, n) in sec.iter().enumerate() {
                if op(n) == "recurrence" {
                    self.recs.insert((s as u8, i), Rec::compile(n)?);
                }
            }
        }
        Ok(())
    }

    pub fn slot(&self, name: &str) -> Option<&Json> {
        self.slots.iter().find(|s| s.get("name").and_then(|n| n.as_str()) == Some(name))
    }

    /// A tensor name with `{i}` replaced for application `app` (IR_SPEC 3.7, 7.1, 7.2).
    pub fn tname(&self, node: &Json, name: &str, app: Option<usize>) -> String {
        match app {
            Some(i) if name.contains("{i}") => {
                let v = if node.get("tensor_index").and_then(|t| t.as_str()) == Some("application") {
                    i
                } else {
                    self.weight_index.get(i).copied().unwrap_or(i)
                };
                name.replace("{i}", &v.to_string())
            }
            _ => name.to_string(),
        }
    }

    pub fn w(&self, name: &str) -> Result<&Tensor, String> {
        self.weights.get(name).ok_or(format!("tensor {} is not a tensor of the file", name))
    }

    /// The tensor a node's attribute names, at application `app`.
    pub fn wa(&self, node: &Json, key: &str, app: Option<usize>) -> Result<&Tensor, String> {
        let n = node.get(key).and_then(|v| v.as_str()).ok_or(format!("{}: no tensor {}", op(node), key))?;
        self.w(&self.tname(node, n, app))
    }

    pub fn wa_opt(&self, node: &Json, key: &str, app: Option<usize>) -> Result<Option<&Tensor>, String> {
        match node.get(key) {
            None | Some(Json::Null) => Ok(None),
            Some(Json::Str(n)) => self.w(&self.tname(node, n, app)).map(Some),
            Some(_) => Err(format!("{}: {} is not a tensor name", op(node), key)),
        }
    }
}
