fn main() {
    // Try pkg-config first (works when spglib is installed system-wide or via a package manager).
    if pkg_config::probe_library("spglib").is_ok() {
        return;
    }

    // Fallback: assume spglib is installed in the standard system paths.
    println!("cargo:rustc-link-lib=spglib");
}
