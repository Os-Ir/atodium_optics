use ash::vk::DeviceSize;
use std::ffi::CStr;

pub mod descriptor_set;
pub mod device;
pub mod shader_compiler;
pub mod shader_reflection;
pub mod pipeline;

pub const VALIDATION_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];

pub const DEVICE_EXTENSIONS: [&CStr; 9] = [
    ash::khr::synchronization2::NAME,
    ash::khr::maintenance4::NAME,
    ash::khr::acceleration_structure::NAME,
    ash::khr::ray_tracing_pipeline::NAME,
    ash::khr::buffer_device_address::NAME,
    ash::khr::deferred_host_operations::NAME,
    ash::khr::shader_float_controls::NAME,
    ash::khr::spirv_1_4::NAME,
    ash::ext::descriptor_indexing::NAME,
];

pub fn align_up(value: DeviceSize, alignment: DeviceSize) -> DeviceSize {
    assert!(alignment.is_power_of_two(), "Alignment must be a power of two");

    (value + alignment - 1) & !(alignment - 1)
}
