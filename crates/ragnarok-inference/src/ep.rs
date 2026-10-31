//! ONNX Runtime execution-provider selection.
//!
//! `inference.provider` accepts `auto` (default), `cpu`, or an explicit
//! provider name (`webgpu`, `vulkan`, `coreml`, `cuda`, `directml`).
//!
//! In `auto` mode every compiled-in GPU provider is registered first —
//! CoreML on Apple Silicon, then the WebGPU EP on Dawn (the **Vulkan**
//! backend on Linux/Windows, Metal on macOS). Registrations fail
//! *silently* (provider libs missing → warn → CPU). An explicit name is
//! hard-checked instead: not-compiled or runtime failures surface as
//! errors so a misconfigured `provider = "vulkan"` can never silently
//! degrade to CPU.

use ort::ep::ExecutionProviderDispatch;
use ragnarok_core::{Error, Result};

/// Process-wide ONNX Runtime environment — ORT allows exactly one per
/// process, so the embedder and reranker sessions must share this
/// singleton instead of calling `ort::init().build()` each.
pub(crate) fn environment() -> Result<ort::environment::Environment> {
    static ENV: std::sync::OnceLock<std::result::Result<ort::environment::Environment, String>> =
        std::sync::OnceLock::new();
    ENV.get_or_init(|| ort::init().build().map_err(|e| format!("ort env: {e}")))
        .clone()
        .map_err(Error::Inference)
}

/// WebGPU EP is compiled into this binary (target has a prebuilt dist).
const HAS_WEBGPU: bool = cfg!(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "windows", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64"),
));
/// CoreML EP (Apple Silicon only — other macOS targets have no dist).
const HAS_COREML: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

fn webgpu_dispatch() -> Option<ExecutionProviderDispatch> {
    #[cfg(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64"),
    ))]
    return Some(ort::ep::WebGPU::default().build());
    #[cfg(not(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64"),
    )))]
    None
}

fn vulkan_dispatch() -> Option<ExecutionProviderDispatch> {
    #[cfg(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64"),
    ))]
    return Some(
        ort::ep::WebGPU::default()
            .with_dawn_backend_type(ort::ep::webgpu::DawnBackendType::Vulkan)
            .build(),
    );
    #[cfg(not(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64"),
    )))]
    None
}

fn coreml_dispatch() -> Option<ExecutionProviderDispatch> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    return Some(ort::ep::CoreML::default().build());
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    None
}

/// Ordered provider chain for `auto`: every compiled-in GPU provider,
/// `.fail_silently()` — ORT falls through to CPU when none registers.
fn auto_chain() -> Vec<ExecutionProviderDispatch> {
    let mut chain = Vec::new();
    if let Some(ep) = coreml_dispatch() {
        chain.push(ep.fail_silently());
    }
    if let Some(ep) = webgpu_dispatch() {
        chain.push(ep.fail_silently());
    }
    chain
}

/// Provider names compiled into this binary (for error messages/logging).
pub fn compiled_providers() -> &'static [&'static str] {
    match (HAS_COREML, HAS_WEBGPU) {
        (true, true) => &["cpu", "coreml", "webgpu", "vulkan"],
        (false, true) => &["cpu", "webgpu", "vulkan"],
        _ => &["cpu"],
    }
}

/// Resolve `inference.provider` into an ordered dispatch list for
/// `SessionBuilder::with_execution_providers`.
pub fn execution_providers(requested: &str) -> Result<Vec<ExecutionProviderDispatch>> {
    match requested.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => Ok(auto_chain()),
        "cpu" => Ok(Vec::new()),
        "webgpu" => webgpu_dispatch()
            .map(|e| vec![e.error_on_failure()])
            .ok_or_else(|| unsupported("webgpu")),
        // Vulkan = WebGPU EP with the Dawn Vulkan backend pinned.
        "vulkan" => vulkan_dispatch()
            .map(|e| vec![e.error_on_failure()])
            .ok_or_else(|| unsupported("vulkan")),
        "coreml" => coreml_dispatch()
            .map(|e| vec![e.error_on_failure()])
            .ok_or_else(|| unsupported("coreml")),
        "cuda" | "tensorrt" | "directml" | "nvrtx" => Err(unsupported(requested)),
        other => Err(Error::Inference(format!(
            "unknown inference.provider {other:?} — expected one of auto, {}",
            compiled_providers().join(", ")
        ))),
    }
}

fn unsupported(name: &str) -> Error {
    Error::Inference(format!(
        "inference.provider {name:?} is not compiled into this binary — \
         available: {}",
        compiled_providers().join(", ")
    ))
}

/// Light-touch hardware hints used for logging — the real probe is EP
/// registration itself (Dawn enumerates adapters internally).
pub fn gpu_hints() -> Vec<&'static str> {
    let mut hints = Vec::new();
    if HAS_COREML {
        hints.push("apple-silicon");
    }
    if nvidia_present() {
        hints.push("nvidia");
    }
    if vulkan_driver_present() {
        hints.push("vulkan-driver");
    }
    hints
}

fn nvidia_present() -> bool {
    std::path::Path::new("/dev/nvidiactl").exists()
        || std::path::Path::new("/proc/driver/nvidia").exists()
        || which("nvidia-smi")
}

fn vulkan_driver_present() -> bool {
    for dir in [
        "/usr/share/vulkan/icd.d",
        "/etc/vulkan/icd.d",
        "/usr/local/share/vulkan/icd.d",
    ] {
        if std::fs::read_dir(dir)
            .map(|mut d| d.next().is_some())
            .unwrap_or(false)
        {
            return true;
        }
    }
    which("vulkaninfo")
}

fn which(bin: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|p| p.join(bin).is_file()))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_returns_fail_silently_gpu_chain() {
        let chain = execution_providers("auto").unwrap();
        if HAS_WEBGPU || HAS_COREML {
            assert!(!chain.is_empty());
        }
        for ep in &chain {
            assert_ne!(ep.name(), "");
        }
    }

    #[test]
    fn cpu_means_no_providers() {
        assert!(execution_providers("cpu").unwrap().is_empty());
    }

    #[cfg(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64"),
    ))]
    #[test]
    fn vulkan_maps_to_webgpu_ep() {
        let chain = execution_providers("vulkan").unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].name(), "WebGpuExecutionProvider");
    }

    #[cfg(not(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64"),
    )))]
    #[test]
    fn vulkan_errors_when_not_compiled() {
        assert!(execution_providers("vulkan").is_err());
    }

    #[test]
    fn unknown_and_uncompiled_providers_error() {
        assert!(execution_providers("bogus").is_err());
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        assert!(execution_providers("cuda").is_err());
    }
}
