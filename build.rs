//! Binary link flags.
//!
//! `ort`'s WebGPU/CUDA execution providers are `NEEDED` dynamic libraries
//! (ONNX Runtime itself is linked statically). `copy-dylibs` drops them
//! next to the binary, so an `$ORIGIN` rpath lets the daemon, tests and an
//! installed binary resolve them without `LD_LIBRARY_PATH`.

fn main() {
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("linux") => println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN"),
        Ok("macos") => println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path"),
        _ => {}
    }
}
