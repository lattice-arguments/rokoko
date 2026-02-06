fn main() {
    // When the "rust-hexl" feature is enabled (default), everything is
    // pure Rust — no external linking required.
    //
    // When "rust-hexl" is disabled, we fall back to the C++ Intel HEXL
    // shared library built via `make hexl && make wrapper`.
    #[cfg(not(feature = "rust-hexl"))]
    {
        println!("cargo:rustc-link-lib=dylib=hexl_wrapper");
        println!("cargo:rustc-link-search=native=.");
        println!("cargo:rustc-link-search=native=./hexl-bindings/hexl/build/hexl/lib");
        println!("cargo:rustc-link-search=native=./hexl-bindings/hexl/build/hexl/lib64");
    }
}
