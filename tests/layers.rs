//! Checks that each layer of the library imports only from the layers below it: `ring` <
//! `common` < `sumcheck` < `commitment` < `rok` < `tools` < the parties (`committer`,
//! `prover`, `verifier`), with `tracing` beside them. Only `ring` calls the
//! `incomplete-rexl` kernels. Test code is exempt, since tests may use helpers from any layer.
//!
//! The check reads the sources: `crate::` and `$crate::` paths, including grouped
//! `use crate::{...}` trees, outside comments, string literals and `#[cfg(test)]` items, and
//! every `mod` declaration in `lib.rs`. It does not follow `super::` paths, aliases of the
//! crate root (`use crate as c`) or `#[macro_export]` macros invoked by bare name, and it treats
//! as production code a test-only module in its own file and items under `cfg(any(test, ...))`,
//! so those can fail it spuriously.

use std::fs;
use std::path::{Path, PathBuf};

/// Each top-level module of the library, lowest layer first, with the modules it may import.
const LAYERS: &[(&str, &[&str])] = &[
    ("ring", &["ring", "tracing"]),
    ("common", &["ring", "common", "tracing"]),
    ("sumcheck", &["ring", "common", "sumcheck", "tracing"]),
    ("commitment", &["ring", "common", "sumcheck", "commitment", "tracing"]),
    ("rok", &["ring", "common", "sumcheck", "commitment", "rok", "tracing"]),
    ("tools", &["ring", "common", "sumcheck", "commitment", "rok", "tools", "tracing"]),
    ("committer", &["ring", "common", "sumcheck", "commitment", "rok", "tools", "committer", "prover", "verifier", "tracing"]),
    ("prover", &["ring", "common", "sumcheck", "commitment", "rok", "tools", "committer", "prover", "verifier", "tracing"]),
    ("verifier", &["ring", "common", "sumcheck", "commitment", "rok", "tools", "committer", "prover", "verifier", "tracing"]),
    ("tracing", &["tracing"]),
];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The number of `#`s if a raw string literal (`r"`, `r#"`, `br"`, `cr"`, ...) starts at `c[i]`.
fn raw_string_start(c: &[char], i: usize) -> Option<usize> {
    let r = match c[i] {
        'r' => i,
        'b' | 'c' if c.get(i + 1) == Some(&'r') => i + 1,
        _ => return None,
    };
    if i > 0 && is_ident(c[i - 1]) {
        return None;
    }
    let hashes = c[r + 1..].iter().take_while(|&&ch| ch == '#').count();
    (c.get(r + 1 + hashes) == Some(&'"')).then_some(hashes)
}

/// The source with comments removed and string and char literals emptied.
fn strip_comments_and_literals(src: &str) -> String {
    let c: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < c.len() {
        if c[i] == '/' && c.get(i + 1) == Some(&'/') {
            while i < c.len() && c[i] != '\n' {
                i += 1;
            }
        } else if c[i] == '/' && c.get(i + 1) == Some(&'*') {
            let mut depth = 0;
            loop {
                if c[i] == '/' && c.get(i + 1) == Some(&'*') {
                    depth += 1;
                    i += 2;
                } else if c[i] == '*' && c.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    i += 1;
                }
            }
        } else if let Some(hashes) = raw_string_start(&c, i) {
            // `r"..."`, `r#"..."#`, `br"..."`: no escapes; ends at `"` and the same hashes.
            i += c[i..].iter().position(|&ch| ch == '"').unwrap() + 1;
            loop {
                if c[i] == '"' && (1..=hashes).all(|k| c.get(i + k) == Some(&'#')) {
                    i += 1 + hashes;
                    break;
                }
                i += 1;
            }
            out.push_str("\"\"");
        } else if c[i] == '"' {
            i += 1;
            while c[i] != '"' {
                i += if c[i] == '\\' { 2 } else { 1 };
            }
            i += 1;
            out.push_str("\"\"");
        } else if c[i] == '\'' && c.get(i + 2) == Some(&'\'') {
            i += 3;
        } else if c[i] == '\'' && c.get(i + 1) == Some(&'\\') {
            // Skip the quote, the backslash and the escaped character, then up to the close.
            i += 3;
            while c[i] != '\'' {
                i += 1;
            }
            i += 1;
        } else {
            out.push(c[i]);
            i += 1;
        }
    }
    out
}

/// The length of the item, field, variant or arm that starts `src`: up to a `;` or `,` outside
/// brackets, through the `}` that closes its body, or up to a closing bracket that ends the list
/// it sits in.
fn item_len(src: &str) -> usize {
    // `angle` counts the generic brackets of a signature or type outside other brackets, so a
    // comma inside `<...>` does not end the item; the `>` of `->` and `=>` is not one.
    let (mut depth, mut angle) = (0usize, 0usize);
    let mut prev = ' ';
    for (k, ch) in src.char_indices() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' if depth == 0 => return k,
            ')' | ']' => depth -= 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return k + 1;
                }
            }
            '<' if depth == 0 => angle += 1,
            '>' if depth == 0 && prev != '-' && prev != '=' => angle = angle.saturating_sub(1),
            ';' if depth == 0 => return k + 1,
            ',' if depth == 0 && angle == 0 => return k + 1,
            _ => {}
        }
        prev = ch;
    }
    src.len()
}

/// The source with everything marked `#[cfg(test)]` removed.
fn strip_test_items(src: &str) -> String {
    const ATTR: &str = "#[cfg(test)]";
    let mut out = String::new();
    let mut rest = src;
    while let Some(at) = rest.find(ATTR) {
        out.push_str(&rest[..at]);
        let item = &rest[at + ATTR.len()..];
        rest = &item[item_len(item)..];
    }
    out.push_str(rest);
    out
}

fn is_ident(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

/// The top-level modules named by `crate::` paths in the (stripped) source.
fn imported_modules(src: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut rest = src;
    while let Some(at) = rest.find("crate::") {
        let before = rest[..at].chars().next_back();
        rest = &rest[at + "crate::".len()..];
        if before.is_some_and(is_ident) {
            continue;
        }
        if let Some(group) = rest.strip_prefix('{') {
            let (mut depth, mut expect) = (1, true);
            let mut chars = group.char_indices().peekable();
            while let Some((k, ch)) = chars.next() {
                match ch {
                    '{' => {
                        depth += 1;
                        expect = false;
                    }
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    ',' if depth == 1 => expect = true,
                    ch if is_ident(ch) && expect && depth == 1 => {
                        let name: String =
                            group[k..].chars().take_while(|&ch| is_ident(ch)).collect();
                        for _ in 1..name.chars().count() {
                            chars.next();
                        }
                        found.push(name);
                        expect = false;
                    }
                    _ => {}
                }
            }
        } else {
            found.push(rest.chars().take_while(|&ch| is_ident(ch)).collect());
        }
    }
    found
}

fn mentions(src: &str, ident: &str) -> bool {
    src.match_indices(ident).any(|(at, _)| {
        let before = src[..at].chars().next_back();
        let after = src[at + ident.len()..].chars().next();
        !before.is_some_and(is_ident) && !after.is_some_and(is_ident)
    })
}

fn violations(layer: &str, allowed: &[&str], src: &str) -> Vec<String> {
    let src = strip_test_items(&strip_comments_and_literals(src));
    let mut found: Vec<String> = imported_modules(&src)
        .into_iter()
        .filter(|m| !allowed.contains(&m.as_str()))
        .map(|m| format!("imports crate::{m}"))
        .collect();
    if layer != "ring" && mentions(&src, "incomplete_rexl") {
        found.push("calls incomplete_rexl outside ring".to_string());
    }
    found
}

/// The modules a (stripped) source declares with `mod NAME`, whatever their visibility.
fn declared_modules(src: &str) -> Vec<String> {
    let depth = |at: usize| src[..at].matches('{').count() - src[..at].matches('}').count();
    src.match_indices("mod")
        .filter(|&(at, _)| depth(at) == 0)
        .filter(|&(at, _)| !src[..at].chars().next_back().is_some_and(is_ident))
        .filter_map(|(at, _)| {
            let after = &src[at + "mod".len()..];
            let name = after.trim_start();
            (name.len() < after.len()).then(|| name.chars().take_while(|&ch| is_ident(ch)).collect())
        })
        .filter(|name: &String| !name.is_empty())
        .collect()
}

#[test]
fn every_top_level_module_has_a_layer() {
    let lib = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs")).unwrap();
    let lib = strip_test_items(&strip_comments_and_literals(&lib));
    let declared = declared_modules(&lib);
    assert!(!declared.is_empty(), "found no module declarations in src/lib.rs");
    for name in declared {
        assert!(
            LAYERS.iter().any(|(layer, _)| *layer == name),
            "src/lib.rs declares `{name}`, which tests/layers.rs assigns to no layer"
        );
    }
}

#[test]
fn layers_import_only_from_below() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut report = Vec::new();
    for (layer, allowed) in LAYERS {
        let mut files = Vec::new();
        // A top-level module is a file `src/<layer>.rs`, a folder `src/<layer>/`, or both.
        let file = src.join(format!("{layer}.rs"));
        if file.is_file() {
            files.push(file);
        }
        if src.join(layer).is_dir() {
            rust_files(&src.join(layer), &mut files);
        }
        assert!(!files.is_empty(), "no sources found for layer `{layer}`");
        for file in files {
            for v in violations(layer, allowed, &fs::read_to_string(&file).unwrap()) {
                report.push(format!("{}: {v}", file.strip_prefix(&src).unwrap().display()));
            }
        }
    }
    assert!(report.is_empty(), "layering violations:\n{}", report.join("\n"));
}

#[test]
fn the_check_finds_what_it_should() {
    // Fixed allowed lists, so the check is tested independently of the layer table.
    let ring: &[&str] = &["ring", "tracing"];
    let common: &[&str] = &["ring", "common", "tracing"];
    let src = r##"
use crate::ring::config::DEGREE;
use crate::{
    ring::helpers::ONE,
    rok::{config::Config, wire},
};
fn f() -> usize { crate::sumcheck::linear::X + incomplete_rexl::y() }
// crate::common in a comment
const S: &str = "crate::common";
const R: &str = r#"crate::common "quoted" \"#;
const C: char = '"';
pub struct T { #[cfg(test)] calls: u32, y: u32 }
enum E { A, #[cfg(test)] B, C }
impl T { fn g() { crate::common::norms::f(); } }
macro_rules! m { () => { $crate::rok::w() }; }
#[cfg(test)]
mod tests {
    use crate::common::init_common;
}
#[cfg(test)]
use crate::common::hash::HashWrapper;
#[cfg(test)]
fn t() -> [u8; 4] { crate::sumcheck::x() }
#[cfg(test)]
fn helper<A, B>(a: A, b: B) { crate::rok::only_in_test(); }
#[cfg(test)]
impl<A, B> Foo<A, B> { fn f() { crate::rok::only_in_test(); } }
#[cfg(test)]
fn h() -> std::collections::HashMap<u8, u8> { crate::rok::only_in_test() }
pub struct U { #[cfg(test)] m: HashMap<u8, crate::rok::X>, y: u8 }
const CR: &CStr = cr#"a"b"#;
"##;
    assert_eq!(
        violations("ring", ring, src),
        vec![
            "imports crate::rok",
            "imports crate::sumcheck",
            "imports crate::common",
            "imports crate::rok"
        ]
    );
    assert_eq!(
        violations("common", common, src),
        vec![
            "imports crate::rok",
            "imports crate::sumcheck",
            "imports crate::rok",
            "calls incomplete_rexl outside ring"
        ]
    );
    let lib = "pub mod a;\nmod b;\npub(crate) mod c;\n#[allow(dead_code)] pub mod d { }\n\
               // mod e;\n#[cfg(test)]\nmod f;\nfn model() {}\npub mod g { mod inner; }\n";
    assert_eq!(
        declared_modules(&strip_test_items(&strip_comments_and_literals(lib))),
        vec!["a", "b", "c", "d", "g"]
    );
}
