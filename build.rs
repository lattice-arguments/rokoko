fn main() {
    let hexl_path = ".";

    println!("cargo:rustc-link-lib=dylib=hexl_wrapper"); // Link to the shared library
    println!("cargo:rustc-link-search=native=.");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", hexl_path);
}
