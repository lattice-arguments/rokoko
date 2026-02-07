fn main() {
    // When the "rust-hexl" feature is enabled (default), everything is
    // pure Rust — no external linking required.
    //
    // When "rust-hexl" is disabled, we fall back to the C++ Intel HEXL
    // shared library built via `make hexl && make wrapper`.
    #[cfg(not(feature = "rust-hexl"))]
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
}
