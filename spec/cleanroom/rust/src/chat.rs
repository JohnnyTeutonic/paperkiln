//! The chat profile (IR_SPEC 8.1): a conversation rendered under a template record (8.1.1, 8.1.3, and
//! 8.1.6's `paperkiln/2` with tool turns), and a generated reply split into content and tool calls
//! (8.1.6, `parse_reply`). Written from the text of v1.4.2.

use crate::json::Json;

/// A refusal of section 8.1 (the adapter's `{"refused": reason}`).
pub type Refused = String;

const TOP: [&str; 19] = [
    "name",
    "version",
    "bos",
    "content",
    "system",
    "user",
    "assistant",
    "generation_prompt",
    "generation_prompt_nothink",
    "stop",
    "past_assistant",
    "default_system",
    "reserved",
    "markers",
    "match",
    "architectures",
    "tool",
    "tool_call",
    "tools",
];

struct Pair {
    open: String,
    close: String,
}

enum SysMode {
    Fold(String),
    Turn(Pair),
    None,
}

struct Record {
    bos: String,
    trim: bool,
    system: SysMode,
    user: Pair,
    assistant: Pair,
    generation_prompt: String,
    nothink: Option<String>,
    strip_think: bool,
    default_system: Option<String>,
    reserved: Vec<String>,
    /// paperkiln/2's three fields (8.1.6)
    tool: Option<Pair>,
    tool_call: Option<(String, String)>,
    tools: Option<(String, String)>,
}

fn members_only(o: &Json, what: &str, allowed: &[&str]) -> Result<(), Refused> {
    match o {
        Json::Obj(m) => {
            for (k, _) in m {
                if !allowed.contains(&k.as_str()) {
                    return Err(format!("record field {}.{} is not implemented", what, k));
                }
            }
            Ok(())
        }
        _ => Err(format!("record field {} is not an object", what)),
    }
}

fn get_str(o: &Json, k: &str, what: &str) -> Result<String, Refused> {
    o.get(k).and_then(|x| x.as_str()).map(String::from).ok_or(format!("record field {}.{} is missing or not a string", what, k))
}

fn pair(r: &Json, k: &str, second: &str) -> Result<Option<(String, String)>, Refused> {
    match r.get(k) {
        None => Ok(None),
        Some(o) => {
            members_only(o, k, &["open", second])?;
            Ok(Some((get_str(o, "open", k)?, get_str(o, second, k)?)))
        }
    }
}

fn record(r: &Json) -> Result<Record, Refused> {
    let m = r.as_obj().ok_or("the record is not an object")?;
    for (k, _) in m {
        if !TOP.contains(&k.as_str()) {
            return Err(format!("record field {} is not implemented", k));
        }
    }
    let st = |k: &str| -> Result<String, Refused> { r.get(k).and_then(|x| x.as_str()).map(String::from).ok_or(format!("record field {} is missing or not a string", k)) };
    let opt_st = |k: &str| -> Result<Option<String>, Refused> {
        match r.get(k) {
            None | Some(Json::Null) => Ok(None),
            Some(Json::Str(s)) => Ok(Some(s.clone())),
            Some(_) => Err(format!("record field {} is not a string", k)),
        }
    };
    // 8.1.3 "Required fields": name, version, user, assistant, generation_prompt and stop; the others
    // default (bos "", content raw, system mode none, past_assistant keep, reserved none)
    for k in ["name", "version", "user", "assistant", "generation_prompt", "stop"] {
        if r.get(k).is_none() {
            return Err(format!("record field {} is missing", k));
        }
    }
    let st_or = |k: &str, d: &str| -> Result<String, Refused> {
        match r.get(k) {
            None => Ok(d.to_string()),
            Some(Json::Str(s)) => Ok(s.clone()),
            Some(_) => Err(format!("record field {} is not a string", k)),
        }
    };
    let trim = match st_or("content", "raw")?.as_str() {
        "raw" => false,
        "trim" => true,
        v => return Err(format!("record field content = {:?} is not implemented", v)),
    };
    // members of system other than mode, open, close and join are not refused (8.1.3)
    let system = match r.get("system") {
        None => SysMode::None,
        Some(sys) => {
            if !matches!(sys, Json::Obj(_)) {
                return Err("record field system is not an object".into());
            }
            match sys.get("mode").and_then(|x| x.as_str()) {
                Some("fold") => SysMode::Fold(get_str(sys, "join", "system")?),
                Some("turn") => SysMode::Turn(Pair { open: get_str(sys, "open", "system")?, close: get_str(sys, "close", "system")? }),
                Some("none") => SysMode::None,
                v => return Err(format!("record field system.mode = {:?} is not implemented", v)),
            }
        }
    };
    let role = |k: &str| -> Result<Pair, Refused> {
        match pair(r, k, "close")? {
            Some((open, close)) => Ok(Pair { open, close }),
            None => Err(format!("record field {} is missing", k)),
        }
    };
    let strip_think = match st_or("past_assistant", "keep")?.as_str() {
        "keep" => false,
        "strip_think" => true,
        v => return Err(format!("record field past_assistant = {:?} is not implemented", v)),
    };
    let reserved: Vec<String> = match r.get("reserved") {
        None => vec![],
        Some(Json::Arr(a)) => a.iter().map(|x| x.as_str().map(String::from)).collect::<Option<Vec<_>>>().ok_or("record field reserved holds a non-string")?,
        Some(_) => return Err("record field reserved is not a list".into()),
    };
    // stop: members other than eos and turn are not refused (8.1.3); it must be an object
    if !matches!(r.get("stop"), Some(Json::Obj(_))) {
        return Err("record field stop is not an object".into());
    }
    let tool = pair(r, "tool", "close")?.map(|(open, close)| Pair { open, close });
    let tool_call = pair(r, "tool_call", "join")?;
    let tools = match r.get("tools") {
        None => None,
        Some(o) => {
            members_only(o, "tools", &["join", "sep"])?;
            Some((get_str(o, "join", "tools")?, get_str(o, "sep", "tools")?))
        }
    };
    // 8.1.6: tool without tool_call or the reverse, tools without them, tool with a system mode other than turn
    if tool.is_some() != tool_call.is_some() {
        return Err("record field tool without tool_call, or the reverse".into());
    }
    if tools.is_some() && tool.is_none() {
        return Err("record field tools without tool and tool_call".into());
    }
    if tool.is_some() && !matches!(system, SysMode::Turn(_)) {
        return Err("record field tool with a system mode other than turn".into());
    }
    Ok(Record {
        bos: st_or("bos", "")?,
        trim,
        system,
        user: role("user")?,
        assistant: role("assistant")?,
        generation_prompt: st("generation_prompt")?,
        nothink: opt_st("generation_prompt_nothink")?,
        strip_think,
        default_system: opt_st("default_system")?,
        reserved,
        tool,
        tool_call,
        tools,
    })
}

/// 8.1.3's whitespace of `trim`: Python's str.strip() and HF's Jinja trim.
fn is_ws(c: char) -> bool {
    matches!(c as u32, 0x09..=0x0D | 0x1C..=0x20 | 0x85 | 0xA0 | 0x1680 | 0x2000..=0x200A | 0x2028 | 0x2029 | 0x202F | 0x205F | 0x3000)
}

/// A definition's text (8.1.6): Python's json.dumps(d, ensure_ascii=False).
fn dumps(v: &Json, out: &mut String) -> Result<(), Refused> {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(n) => {
            if !n.int_form {
                return Err("a tool definition holds a number written with a fraction or an exponent".into());
            }
            if n.v.abs() < 9.0e15 {
                out.push_str(&format!("{}", n.v as i64));
            } else {
                return Err("a tool definition holds an integer beyond this engine's range".into());
            }
        }
        Json::Str(s) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\u{8}' => out.push_str("\\b"),
                    '\t' => out.push_str("\\t"),
                    '\n' => out.push_str("\\n"),
                    '\u{c}' => out.push_str("\\f"),
                    '\r' => out.push_str("\\r"),
                    c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        Json::Arr(a) => {
            out.push('[');
            for (k, x) in a.iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                dumps(x, out)?;
            }
            out.push(']');
        }
        Json::Obj(m) => {
            out.push('{');
            for (k, (key, x)) in m.iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                dumps(&Json::Str(key.clone()), out)?;
                out.push_str(": ");
                dumps(x, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

struct Call {
    id: Option<String>,
    name: String,
    arguments: String,
}

struct Msg {
    role: String,
    content: String,
    calls: Vec<Call>,
    /// absent, or present: Some(None) for null
    tool_call_id: Option<Option<String>>,
}

fn calls_of(m: &Json, i: usize) -> Result<Vec<Call>, Refused> {
    let tc = match m.get("tool_calls") {
        None | Some(Json::Null) => return Ok(vec![]),
        Some(Json::Arr(a)) if a.is_empty() => return Err(format!("message {}: an empty tool_calls list", i)),
        Some(Json::Arr(a)) => a,
        Some(_) => return Err(format!("message {}: tool_calls is not a list", i)),
    };
    let mut out = Vec::new();
    for c in tc {
        if !matches!(c, Json::Obj(_)) {
            return Err(format!("message {}: a call is not an object", i));
        }
        let id = match c.get("id") {
            None => None,
            Some(Json::Str(s)) => Some(s.clone()),
            Some(_) => return Err(format!("message {}: a call's id is not a string", i)),
        };
        match c.get("type") {
            None => {}
            Some(Json::Str(t)) if t == "function" => {}
            Some(_) => return Err(format!("message {}: a call's type is not \"function\"", i)),
        }
        let f = c.get("function").filter(|f| matches!(f, Json::Obj(_))).ok_or(format!("message {}: a call has no function object", i))?;
        let name = match f.get("name") {
            Some(Json::Str(s)) if !s.is_empty() && !s.contains('\n') => s.clone(),
            _ => return Err(format!("message {}: a call's name is not a non-empty string without a newline", i)),
        };
        let arguments = match f.get("arguments") {
            None => String::new(),
            Some(Json::Str(s)) => s.clone(),
            Some(_) => return Err(format!("message {}: a call's arguments are not a string", i)),
        };
        out.push(Call { id, name, arguments });
    }
    Ok(out)
}

/// `render_chat` (11.3): the rendered text, or the refusal.
pub fn render(p: &Json) -> Result<String, Refused> {
    let rec = record(p.get("record").ok_or("no record")?)?;
    let want = p.get("generation_prompt").and_then(|x| x.as_bool()).unwrap_or(false);
    let thinking = p.get("thinking").and_then(|x| x.as_bool());
    let has_tool = rec.tool.is_some();
    // the conversation's tool definitions
    let defs: Vec<String> = match p.get("tools") {
        None | Some(Json::Null) => vec![],
        Some(Json::Arr(a)) => {
            if !a.is_empty() && rec.tools.is_none() {
                return Err("tool definitions under a record without tools".into());
            }
            let mut v = Vec::new();
            for d in a {
                if !matches!(d, Json::Obj(_)) {
                    return Err("a tool definition is not an object".into());
                }
                let mut t = String::new();
                dumps(d, &mut t)?;
                v.push(t);
            }
            v
        }
        Some(_) => return Err("tools is not a list".into()),
    };
    // the messages, each checked against its role's members
    let raw = p.get("messages").and_then(|x| x.as_arr()).ok_or("messages is not a list")?;
    let mut msgs: Vec<Msg> = Vec::new();
    for (i, m) in raw.iter().enumerate() {
        if !matches!(m, Json::Obj(_)) {
            return Err(format!("message {} is not an object", i));
        }
        let role = m.get("role").and_then(|x| x.as_str()).ok_or(format!("message {} has no role", i))?.to_string();
        match role.as_str() {
            "system" | "user" | "assistant" => {}
            "tool" if has_tool => {}
            "tool" => return Err("the tool role under a record without tool".into()),
            r => return Err(format!("message {}: role {:?} is not defined", i, r)),
        }
        match m.get("loss") {
            None => {}
            Some(Json::Bool(_)) if role == "assistant" => {}
            Some(_) => return Err(format!("message {}: loss on a {} message", i, role)),
        }
        // 8.1.1, under any record: a tool_calls member (even null) under a record without tool, or on a
        // message that is not an assistant message, is refused, as is a tool_call_id member on a message
        // that is not a tool message. Other members than role, content and loss are ignored.
        if m.get("tool_calls").is_some() {
            if !has_tool {
                return Err(format!("message {}: tool_calls under a record without tool", i));
            }
            if role != "assistant" {
                return Err(format!("message {}: tool_calls on a {} message", i, role));
            }
        }
        if m.get("tool_call_id").is_some() && role != "tool" {
            return Err(format!("message {}: tool_call_id on a {} message", i, role));
        }
        let calls = if role == "assistant" { calls_of(m, i)? } else { vec![] };
        // on a tool message (8.1.6): absent, or present as a string or null (checked against the call)
        let tool_call_id: Option<Option<String>> = match m.get("tool_call_id") {
            None => None,
            Some(Json::Null) => Some(None),
            Some(Json::Str(s)) => Some(Some(s.clone())),
            Some(_) => return Err(format!("message {}: tool_call_id is neither a string nor null", i)),
        };
        // every message has a string content, an assistant message with calls included (8.1.1)
        let content = match m.get("content") {
            Some(Json::Str(s)) => s.clone(),
            _ => return Err(format!("message {}: content is absent or not a string", i)),
        };
        msgs.push(Msg { role, content, calls, tool_call_id });
    }
    // structure: 8.1.1, or 8.1.6 under a record with tool
    check_structure(&msgs, want, has_tool)?;
    // reserved strings: contents, call names and arguments, definitions
    let reserved_in = |t: &str| rec.reserved.iter().find(|r| !r.is_empty() && t.contains(r.as_str())).cloned();
    // 8.1.3: each content as rendered, after trim (the system text of mode fold untrimmed) and before
    // strip_think
    for (i, m) in msgs.iter().enumerate() {
        let as_rendered = if rec.trim && !(m.role == "system" && matches!(rec.system, SysMode::Fold(_))) {
            m.content.trim_matches(is_ws).to_string()
        } else {
            m.content.clone()
        };
        if let Some(r) = reserved_in(&as_rendered) {
            return Err(format!("message {} holds the reserved string {:?}", i, r));
        }
        for c in &m.calls {
            if let Some(r) = reserved_in(&c.name).or_else(|| reserved_in(&c.arguments)) {
                return Err(format!("message {}: a call holds the reserved string {:?}", i, r));
            }
        }
    }
    for d in &defs {
        if let Some(r) = reserved_in(d) {
            return Err(format!("a tool definition holds the reserved string {:?}", r));
        }
    }
    // rendering (8.1.3, 8.1.6)
    let last_user = msgs.iter().rposition(|m| m.role == "user");
    let content_of = |m: &Msg, i: usize| -> String {
        let mut c = m.content.clone();
        if rec.strip_think && m.role == "assistant" && last_user.map(|u| i < u).unwrap_or(false) && c.contains("</think>") {
            c = c.rsplit("</think>").next().unwrap_or("").trim_start_matches('\n').to_string();
        }
        if rec.trim {
            c = c.trim_matches(is_ws).to_string();
        }
        c
    };
    let mut out = rec.bos.clone();
    let sys_msg = msgs.first().filter(|m| m.role == "system");
    let sys_text: Option<String> = sys_msg.map(|m| m.content.clone()).or(rec.default_system.clone());
    let mut fold_prefix: Option<String> = None;
    match &rec.system {
        SysMode::None => {
            if sys_msg.is_some() {
                return Err("a system message under system mode none".into());
            }
        }
        SysMode::Fold(join) => {
            if let Some(t) = &sys_text {
                fold_prefix = Some(format!("{}{}", t, join));
            }
        }
        SysMode::Turn(pr) => {
            // the system text (trimmed under trim, as each message's own text), then the definitions
            let text = sys_text.map(|t| if rec.trim { t.trim_matches(is_ws).to_string() } else { t });
            let body = match (&text, defs.is_empty()) {
                (Some(t), false) if !t.is_empty() => {
                    let (join, sep) = rec.tools.clone().unwrap_or_default();
                    Some(format!("{}{}{}", t, join, defs.join(&sep)))
                }
                (_, false) => {
                    let (_, sep) = rec.tools.clone().unwrap_or_default();
                    Some(defs.join(&sep))
                }
                (t, true) => t.clone(),
            };
            if let Some(b) = body {
                out.push_str(&pr.open);
                out.push_str(&b);
                out.push_str(&pr.close);
            }
        }
    }
    for (i, m) in msgs.iter().enumerate() {
        match m.role.as_str() {
            "system" => {}
            "user" => {
                let mut c = content_of(m, i);
                if let Some(pfx) = fold_prefix.take() {
                    c = format!("{}{}", pfx, c);
                }
                out.push_str(&rec.user.open);
                out.push_str(&c);
                out.push_str(&rec.user.close);
            }
            "assistant" => {
                out.push_str(&rec.assistant.open);
                out.push_str(&content_of(m, i));
                if let Some((open, join)) = &rec.tool_call {
                    for c in &m.calls {
                        out.push_str(open);
                        out.push_str(&c.name);
                        out.push_str(join);
                        out.push_str(&c.arguments);
                    }
                }
                out.push_str(&rec.assistant.close);
            }
            _ => {
                let t = rec.tool.as_ref().unwrap();
                out.push_str(&t.open);
                out.push_str(&content_of(m, i));
                out.push_str(&t.close);
            }
        }
    }
    if want {
        match (thinking, &rec.nothink) {
            (Some(false), Some(nt)) => out.push_str(nt),
            _ => out.push_str(&rec.generation_prompt),
        }
    }
    Ok(out)
}

fn check_structure(msgs: &[Msg], want: bool, has_tool: bool) -> Result<(), Refused> {
    let body: &[Msg] = if msgs.first().map(|m| m.role == "system").unwrap_or(false) { &msgs[1..] } else { msgs };
    // expected next role, and the calls awaiting results
    let mut pending: Vec<Option<String>> = Vec::new();
    let mut prev: Option<&str> = None;
    for (k, m) in body.iter().enumerate() {
        let ok = match (prev, m.role.as_str()) {
            (_, "system") => false,
            (None, r) => r == "user",
            (Some("user"), r) => r == "assistant",
            (Some("assistant"), r) if !pending.is_empty() => r == "tool",
            (Some("assistant"), r) => r == "user",
            (Some("tool"), r) if !pending.is_empty() => r == "tool",
            (Some("tool"), r) => r == "assistant",
            _ => false,
        };
        if !ok {
            return Err(format!("message {} (role {}) breaks the conversation's structure", k + (msgs.len() - body.len()), m.role));
        }
        if m.role == "assistant" {
            pending = m.calls.iter().rev().map(|c| c.id.clone()).collect();
        }
        if m.role == "tool" {
            let call_id = pending.pop().unwrap();
            // 8.1.6: a present tool_call_id (null included) equals the answered call's id, a call without
            // an id counting as id null
            if let Some(tid) = &m.tool_call_id {
                if *tid != call_id {
                    return Err(format!("tool_call_id {:?} is not the answered call's id {:?} (None: no id)", tid, call_id));
                }
            }
        }
        prev = Some(m.role.as_str());
    }
    let last = body.last().map(|m| m.role.as_str());
    let ends_ok = if want {
        match last {
            Some("user") => true,
            Some("tool") => has_tool && pending.is_empty(),
            _ => false,
        }
    } else {
        last == Some("assistant")
    };
    if !ends_ok {
        return Err(format!(
            "a {} conversation ending on {:?}",
            if want { "prompt" } else { "training" },
            last.unwrap_or(if msgs.is_empty() { "nothing" } else { "the system message" })
        ));
    }
    Ok(())
}

/// `parse_reply` (8.1.6, 11.3).
pub fn parse_reply(p: &Json) -> Result<Json, String> {
    let tokens = p.get("tokens").and_then(|x| x.as_obj()).ok_or("no tokens")?;
    let ids: Vec<i64> = p.get("ids").and_then(|x| x.as_arr()).ok_or("no ids")?.iter().map(|x| x.as_i64().ok_or("an id is not an integer")).collect::<Result<_, _>>()?;
    let call = p.get("call_id").and_then(|x| x.as_i64()).ok_or("no call_id")?;
    let text = |id: i64| -> Result<String, String> {
        let k = id.to_string();
        tokens.iter().rev().find(|(t, _)| *t == k).and_then(|(_, v)| v.as_str().map(String::from)).ok_or(format!("id {} has no text in tokens", id))
    };
    let mut parts: Vec<String> = vec![String::new()];
    for &id in &ids {
        if id == call {
            parts.push(String::new());
        } else {
            let t = text(id)?;
            parts.last_mut().unwrap().push_str(&t);
        }
    }
    let mut calls = Vec::new();
    for (k, part) in parts.iter().enumerate().skip(1) {
        match part.find('\n') {
            Some(at) if at > 0 => {
                let name = &part[..at];
                let arguments = &part[at + 1..];
                calls.push(Json::obj(vec![
                    ("id", Json::str(&format!("call_{}", k - 1))),
                    ("type", Json::str("function")),
                    ("function", Json::obj(vec![("name", Json::str(name)), ("arguments", Json::str(arguments))])),
                ]));
            }
            _ => return Ok(Json::obj(vec![("parsed", Json::Bool(false))])),
        }
    }
    Ok(Json::obj(vec![("parsed", Json::Bool(true)), ("content", Json::str(&parts[0])), ("tool_calls", Json::Arr(calls))]))
}
