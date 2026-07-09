fn main() {
    // When the "incomplete-rexl" feature is enabled (default), everything is
    // pure Rust — no external linking required.
    //
    // When "incomplete-rexl" is disabled, we fall back to the C++ Intel HEXL
    // shared library built via `make hexl && make wrapper`.
    #[cfg(not(feature = "incomplete-rexl"))]
    {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

        println!("cargo:rustc-link-lib=dylib=hexl_wrapper");
        println!("cargo:rustc-link-search=native=.");
        println!("cargo:rustc-link-search=native=./hexl-bindings/hexl/build/hexl/lib");
        println!("cargo:rustc-link-search=native=./hexl-bindings/hexl/build/hexl/lib64");

        // Embed rpath so the binary can find shared libraries at runtime
        // without needing to set LD_LIBRARY_PATH
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", manifest_dir);
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}/hexl-bindings/hexl/build/hexl/lib",
            manifest_dir
        );
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}/hexl-bindings/hexl/build/hexl/lib64",
            manifest_dir
        );
    }

    // Build GIT_SHA env when snapshot profiling is enabled
    #[cfg(feature = "profile")]
    {
        let sha = std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        println!("cargo:rustc-env=GIT_SHA={sha}");
        println!("cargo:rerun-if-changed=.git/HEAD");
        println!("cargo:rerun-if-changed=.git/refs");
    }
}
