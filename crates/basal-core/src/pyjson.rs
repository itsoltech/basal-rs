//! Text of JSON values exactly as upstream basal renders them into prompts:
//! `json.dumps(x, ensure_ascii=False)` with Python's default separators (`", "`, `": "`),
//! Python float `repr` and Python's escaping. Object key order is the request order
//! (serde_json `preserve_order`), numbers keep their literal kind (`arbitrary_precision`).

use serde_json::{Number, Value};

/// upstream `server._text`: strings as they are, everything else as compact Python JSON.
pub fn text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => dumps(other),
    }
}

/// `json.dumps(v, ensure_ascii=False)`.
pub fn dumps(v: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, v);
    out
}

fn write_value(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => out.push_str(&number(n)),
        Value::String(s) => write_str(out, s),
        Value::Array(xs) => {
            out.push('[');
            for (i, x) in xs.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value(out, x);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_str(out, k);
                out.push_str(": ");
                write_value(out, x);
            }
            out.push('}');
        }
    }
}

fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's json module parses a literal with `.`, `e` or `E` as float, otherwise as int.
fn number(n: &Number) -> String {
    let lit = n.to_string();
    if lit.contains(['.', 'e', 'E']) {
        match lit.parse::<f64>() {
            Ok(f) => float_repr(f),
            Err(_) => lit,
        }
    } else if lit.trim_start_matches('-').bytes().all(|b| b == b'0') {
        "0".to_string()
    } else {
        lit
    }
}

/// Python `repr(float)` as used by `json.dumps` (shortest round-trip digits; positional notation for
/// decimal exponents -4..=15, otherwise `d.ddde+XX`).
pub fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let sci = format!("{:e}", f.abs());
    let (mant, exp) = sci.split_once('e').expect("LowerExp has an exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let mut s = String::new();
    if f < 0.0 {
        s.push('-');
    }
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                s.push_str(&digits);
                s.extend(std::iter::repeat_n('0', int_len - digits.len()));
                s.push_str(".0");
            } else {
                s.push_str(&digits[..int_len]);
                s.push('.');
                s.push_str(&digits[int_len..]);
            }
        } else {
            s.push_str("0.");
            s.extend(std::iter::repeat_n('0', (-exp - 1) as usize));
            s.push_str(&digits);
        }
    } else {
        s.push_str(&digits[..1]);
        if digits.len() > 1 {
            s.push('.');
            s.push_str(&digits[1..]);
        }
        s.push_str(&format!("e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs()));
    }
    s
}
