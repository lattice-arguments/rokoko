// Just enough JSON to emit two well-formed artifact files. `serde` + `serde_json`
// carry a derive macro and a full parser/deserializer; profiling only ever writes,
// and only writes these two fixed shapes.

use std::fmt::Write as _;

/// Append `s` as a quoted, escaped JSON string.
pub fn str_into(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
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

/// `"key": "value"` at the given indent, with a trailing comma when `last` is false.
pub fn field_str(out: &mut String, indent: &str, key: &str, value: &str, last: bool) {
    out.push_str(indent);
    str_into(out, key);
    out.push_str(": ");
    str_into(out, value);
    if !last {
        out.push(',');
    }
    out.push('\n');
}
