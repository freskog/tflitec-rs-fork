use std::env;
use std::path::PathBuf;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    println!("cargo:rerun-if-changed=build.rs");

    // Platform-specific linking
    match (target_os.as_str(), target_arch.as_str()) {
        ("linux", "aarch64") => {
            println!("cargo:warning=🔧 Building minimal tflitec for ARM64 Linux");
            link_custom_libraries_linux_aarch64();
        }
        ("macos", _) => {
            println!("cargo:warning=🔧 Building minimal tflitec for macOS");
            link_system_libraries_macos();
        }
        ("windows", _) => {
            println!("cargo:warning=🔧 Building minimal tflitec for Windows");
            link_system_libraries_windows();
        }
        _ => {
            println!(
                "cargo:warning=⚠️  Unsupported platform: {}-{}",
                target_os, target_arch
            );
            println!("cargo:warning=    Falling back to system TensorFlow Lite");
        }
    }
}

/// Link custom TensorFlow Lite libraries for ARM64 Linux (Pi, etc.)
fn link_custom_libraries_linux_aarch64() {
    // Check for custom library path from environment
    if let Ok(custom_lib_path) = env::var("TFLITEC_CUSTOM_LIBRARY_PATH") {
        println!(
            "cargo:warning=📚 Using custom TF Lite library: {}",
            custom_lib_path
        );

        let lib_path = PathBuf::from(&custom_lib_path);
        let lib_dir = lib_path.parent().unwrap();

        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=dylib=tensorflowlite_c");

        // Link supporting XNNPACK libraries if available
        if lib_dir.join("libcpuinfo.so").exists() {
            println!("cargo:rustc-link-lib=dylib=cpuinfo");
            println!("cargo:warning=🔗 Linking libcpuinfo.so");
        }

        if lib_dir.join("libpthreadpool.so").exists() {
            println!("cargo:rustc-link-lib=dylib=pthreadpool");
            println!("cargo:warning=🔗 Linking libpthreadpool.so");
        }

        return;
    }

    // Default: link system libraries
    println!("cargo:warning=📦 No custom library specified, linking system TensorFlow Lite");
    println!("cargo:rustc-link-lib=dylib=tensorflowlite_c");
}

/// Link system TensorFlow Lite libraries for macOS
fn link_system_libraries_macos() {
    println!("cargo:rustc-link-lib=dylib=tensorflowlite_c");

    // Common macOS system frameworks that might be needed
    println!("cargo:rustc-link-lib=framework=Accelerate");
}

/// Link system TensorFlow Lite libraries for Windows  
fn link_system_libraries_windows() {
    println!("cargo:rustc-link-lib=dylib=tensorflowlite_c");
}
