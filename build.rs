use std::env;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

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
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    // Check for prebuilt library first
    let prebuilt_var = match target_arch.as_str() {
        "aarch64" => "TFLITEC_PREBUILT_PATH_AARCH64_APPLE_DARWIN",
        "x86_64" => "TFLITEC_PREBUILT_PATH_X86_64_APPLE_DARWIN",
        _ => "TFLITEC_PREBUILT_PATH",
    };

    if let Ok(prebuilt_path) = env::var(prebuilt_var) {
        let lib_path = PathBuf::from(&prebuilt_path);
        if lib_path.exists() {
            let lib_dir = lib_path.parent().unwrap_or_else(|| Path::new("."));
            println!(
                "cargo:warning=📚 Using prebuilt TensorFlow Lite: {}",
                prebuilt_path
            );
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
            println!("cargo:rustc-link-lib=dylib=tensorflowlite_c");
            println!("cargo:rustc-link-lib=framework=Accelerate");
            return;
        }
    }

    // Check the generic TFLITEC_PREBUILT_PATH as fallback
    if let Ok(prebuilt_path) = env::var("TFLITEC_PREBUILT_PATH") {
        let lib_path = PathBuf::from(&prebuilt_path);
        if lib_path.exists() {
            let lib_dir = lib_path.parent().unwrap_or_else(|| Path::new("."));
            println!(
                "cargo:warning=📚 Using prebuilt TensorFlow Lite (generic): {}",
                prebuilt_path
            );
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
            println!("cargo:rustc-link-lib=dylib=tensorflowlite_c");
            println!("cargo:rustc-link-lib=framework=Accelerate");
            return;
        }
    }

    // Check if TensorFlow Lite is available via pkg-config or homebrew
    if check_tensorflow_lite_available() {
        println!("cargo:warning=📦 Found system TensorFlow Lite, linking...");
        println!("cargo:rustc-link-lib=dylib=tensorflowlite_c");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        return;
    }

    // TensorFlow Lite not found - provide helpful instructions
    println!("cargo:warning=❌ TensorFlow Lite C library not found on macOS!");
    println!("cargo:warning=");
    println!("cargo:warning=To fix this, you have several options:");
    println!("cargo:warning=");
    println!("cargo:warning=1. Install TensorFlow Lite via Homebrew:");
    println!("cargo:warning=   brew install tensorflow-lite");
    println!("cargo:warning=");
    println!("cargo:warning=2. Build and provide a prebuilt library:");
    println!("cargo:warning=   Set environment variable:");
    if target_arch == "aarch64" {
        println!("cargo:warning=   export TFLITEC_PREBUILT_PATH_AARCH64_APPLE_DARWIN=/path/to/libtensorflowlite_c.dylib");
    } else {
        println!("cargo:warning=   export TFLITEC_PREBUILT_PATH_X86_64_APPLE_DARWIN=/path/to/libtensorflowlite_c.dylib");
    }
    println!("cargo:warning=");
    println!("cargo:warning=3. For development, you can skip wakeword functionality by building without it:");
    println!("cargo:warning=   cargo build --release --exclude wakeword");
    println!("cargo:warning=");

    // Don't panic, just emit the link instruction and let the linker fail with a clear error
    println!("cargo:rustc-link-lib=dylib=tensorflowlite_c");
    println!("cargo:rustc-link-lib=framework=Accelerate");
}

/// Check if TensorFlow Lite is available on the system
fn check_tensorflow_lite_available() -> bool {
    // Check with pkg-config first
    if Command::new("pkg-config")
        .args(&["--exists", "tensorflow-lite"])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
    {
        return true;
    }

    // Check common homebrew paths
    let homebrew_paths = [
        "/opt/homebrew/lib/libtensorflowlite_c.dylib", // Apple Silicon
        "/usr/local/lib/libtensorflowlite_c.dylib",    // Intel
    ];

    for path in &homebrew_paths {
        if PathBuf::from(path).exists() {
            return true;
        }
    }

    false
}

/// Link system TensorFlow Lite libraries for Windows  
fn link_system_libraries_windows() {
    println!("cargo:rustc-link-lib=dylib=tensorflowlite_c");
}
