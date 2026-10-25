//! The IR digest (IR_SPEC 3.10): I-JSON checks on the whole text, prose removed, RFC 8785, SHA-256.

use crate::json;
use crate::sha256;

/// (digest, canonical form), or why the text has no digest. `prose` keeps the `_` members
/// (a test of the serialiser alone).
pub fn digest(text: &str, prose: bool) -> Result<(String, String), String> {
    let p = json::parse(text).map_err(|e| format!("not JSON: {}", e))?;
    if let Some(first) = p.ijson_issues.first() {
        return Err(first.clone());
    }
    let v = if prose { p.value } else { json::strip_prose(&p.value) };
    let mut c = String::new();
    json::canonical(&v, &mut c);
    Ok((sha256::hex(&sha256::sha256(c.as_bytes())), c))
}
