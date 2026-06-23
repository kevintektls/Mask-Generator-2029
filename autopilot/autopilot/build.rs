fn main() {
    // depthai-sys stages .so next to the binary; search $ORIGIN at runtime.
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
}
