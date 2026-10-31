//! Probe Vulkan device-local memory before paying the device-lost price
//! of finding out a model can't fit. `libvulkan` is dlopened at call
//! time — hosts without a driver (or with only lavapipe/llvmpipe CPU
//! devices) simply report `None` and the caller keeps its old behavior.

use ash::vk;

/// What the largest usable Vulkan GPU reports — `budget` is the
/// currently-allocatable bytes of the biggest `DEVICE_LOCAL` heap
/// (VK_EXT_memory_budget when the driver answers, heap `size`
/// otherwise). `device` is the Vulkan device name for logging.
pub struct GpuMemory {
    pub device: String,
    pub budget: u64,
}

/// Largest device-local heap budget across all real GPUs. CPU-type
/// devices (llvmpipe/lavapipe software rasterizers) are skipped — a
/// "GPU" that's just the CPU in a trench coat would win the
/// biggest-heap contest on integrated systems otherwise.
pub fn device_local_budget() -> Option<GpuMemory> {
    let entry = unsafe { ash::Entry::load() }.ok()?;
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_0);
    let info = vk::InstanceCreateInfo::default().application_info(&app);
    let instance = unsafe { entry.create_instance(&info, None) }.ok()?;
    let devices = unsafe { instance.enumerate_physical_devices() }.unwrap_or_default();
    let mut best: Option<GpuMemory> = None;
    for pd in devices {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        if props.device_type == vk::PhysicalDeviceType::CPU {
            continue;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut mem2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe { instance.get_physical_device_memory_properties2(pd, &mut mem2) };
        let heaps = &mem2.memory_properties.memory_heaps;
        let n = mem2.memory_properties.memory_heap_count as usize;
        let budget_bytes = (0..n)
            .filter(|&i| heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|i| {
                // VK_EXT_memory_budget answers with a live allocatable
                // figure; drivers without it leave heap_budget zero —
                // fall back to the heap's total size then
                let b = budget.heap_budget[i];
                if b > 0 {
                    b
                } else {
                    heaps[i].size
                }
            })
            .max();
        let Some(bytes) = budget_bytes.filter(|b| *b > 0) else {
            continue;
        };
        if best.as_ref().map(|g| bytes > g.budget).unwrap_or(true) {
            let name = unsafe {
                std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            };
            best = Some(GpuMemory {
                device: name,
                budget: bytes,
            });
        }
    }
    unsafe { instance.destroy_instance(None) };
    best
}

/// Conservative GPU-fit check: the graph file's bytes must leave
/// another like amount for ORT workspace + activation buffers, or
/// sessions die mid-forward (fp32 bge-m3: 2.3 GB file + ~4 GB
/// activations vs a 2 GiB iGPU heap). `None` budget → don't advise.
pub fn model_fits(model_bytes: u64, gpu: Option<&GpuMemory>) -> Option<bool> {
    gpu.map(|g| model_bytes.saturating_mul(2) <= g.budget)
}
