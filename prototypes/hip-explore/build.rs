use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::os::unix::fs::symlink;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);
    
    // We need to link against libhiprtc.so, but on some ROCm installations (like the one present),
    // the symbols are bundled in libamdhip64.so.
    // We attempt to find libamdhip64.so and create a symlink named libhiprtc.so pointing to it.

    let target_lib_name = "libamdhip64.so";
    let link_name = out_path.join("libhiprtc.so");

    // Remove existing link if it exists to avoid error
    if link_name.exists() {
        let _ = std::fs::remove_file(&link_name);
    }

    // 1. Try to find library using ldconfig
    let found_path = find_library_via_ldconfig(target_lib_name)
        // 2. Fallback: Check common paths
        .or_else(|| check_common_paths(target_lib_name));

    if let Some(target_path) = found_path {
        println!("cargo:warning=Found {} at {:?}", target_lib_name, target_path);
        
        if let Err(e) = symlink(&target_path, &link_name) {
             println!("cargo:warning=Failed to create libhiprtc.so symlink: {}", e);
        } else {
             println!("cargo:rustc-link-search=native={}", out_dir);
        }
    } else {
        println!("cargo:warning=Could not find {}. Linking against libhiprtc might fail if it's not in the default path.", target_lib_name);
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LD_LIBRARY_PATH");
}

fn find_library_via_ldconfig(lib_name: &str) -> Option<PathBuf> {
    let output = Command::new("ldconfig")
        .arg("-p")
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Format is usually:
    //  libname.so (libc6,x86-64) => /path/to/libname.so
    
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(lib_name) {
             if let Some(idx) = trimmed.find("=>") {
                 let path_str = trimmed[idx + 2..].trim();
                 let path = PathBuf::from(path_str);
                 if path.exists() {
                     return Some(path);
                 }
             }
        }
    }
    None
}

fn check_common_paths(lib_name: &str) -> Option<PathBuf> {
    let paths = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib",
        "/usr/local/lib",
        "/opt/rocm/lib",
        "/lib/x86_64-linux-gnu",
        "/lib64",
    ];

    for p in paths {
        let path = Path::new(p).join(lib_name);
        if path.exists() {
            return Some(path);
        }
    }
    None
}