use crate::render_resource::render_buffer::{RenderBufferAllocator, RenderBufferAllocatorRef};
use crate::render_resource::render_image::{ImageAllocator, ImageAllocatorRef};
use crate::vk_context::device::{WrappedDevice, WrappedDeviceRef};
use crate::vk_context::shader_compiler::ShaderIncludeStructure;
use anyhow::Result;
use ash::vk;
use ash::vk::DeviceSize;
use std::ffi::CStr;

pub mod descriptor_set;
pub mod device;
pub mod pipeline;
pub mod shader_compiler;
pub mod shader_reflection;
pub mod bindless_descriptor;

pub const ENGINE_NAME: &str = "atodium_optics";
pub const ENGINE_VERSION: u32 = vk::make_api_version(0, 1, 1, 1);

pub const API_VERSION: u32 = vk::API_VERSION_1_3;

pub const VALIDATION_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];

pub const DEVICE_EXTENSIONS: [&CStr; 10] = [
    ash::khr::synchronization2::NAME,
    ash::khr::maintenance4::NAME,
    ash::khr::acceleration_structure::NAME,
    ash::khr::ray_tracing_pipeline::NAME,
    ash::khr::ray_query::NAME,
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

pub fn init_vulkan_context(enable_validation: bool, app_name: &str, app_version: u32) -> Result<(WrappedDeviceRef, RenderBufferAllocatorRef, ImageAllocatorRef, ShaderIncludeStructure)> {
    let device: WrappedDeviceRef = WrappedDevice::new(
        enable_validation,
        &VALIDATION_LAYERS,
        ENGINE_NAME,
        ENGINE_VERSION,
        app_name,
        app_version,
        API_VERSION,
        &DEVICE_EXTENSIONS,
    )?
    .into();

    let buffer_allocator: RenderBufferAllocatorRef = RenderBufferAllocator::new(device.clone())?.into();
    let image_allocator: ImageAllocatorRef = ImageAllocator::new(device.clone(), buffer_allocator.clone()).into();

    let include_structure = shader_compiler::load_shaders();

    Ok((device, buffer_allocator, image_allocator, include_structure))
}
