//! Checks that every source file of the library and the binary opens with a module doc (`//!`)
//! saying what the file is for, so that a new file cannot be added without one.

use std::fs;
use std::path::{Path, PathBuf};

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

#[test]
fn every_source_file_opens_with_a_module_doc() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files);
    assert!(!files.is_empty(), "found no sources under src/");
    let mut missing: Vec<String> = files
        .iter()
        .filter(|file| !fs::read_to_string(file).unwrap().starts_with("//!"))
        .map(|file| file.strip_prefix(&src).unwrap().display().to_string())
        .collect();
    missing.sort();
    assert!(
        missing.is_empty(),
        "these files do not open with a `//!` module doc:\n{}",
        missing.join("\n")
    );
}
