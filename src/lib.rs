use crate::render_resource::render_buffer::{RenderBufferAllocator, RenderBufferAllocatorRef};
use crate::vulkan_context::descriptor_set::{DescriptorId, WrappedDescriptorSet};
use crate::vulkan_context::device::{WrappedDevice, WrappedDeviceRef};
use crate::vulkan_context::pipeline::{PipelineDesc, WrappedPipeline};
use crate::vulkan_context::{shader_compiler, API_VERSION, DEVICE_EXTENSIONS, VALIDATION_LAYERS};
use anyhow::{anyhow, Result};
use ash::vk::BufferUsageFlags;
use gpu_allocator::MemoryLocation;
use image::{ImageBuffer, ImageFormat};
use std::ffi::{c_char, CStr};
use std::mem;
use std::path::Path;

pub mod render_resource;
pub mod vulkan_context;

#[inline]
pub fn lib_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

#[inline]
pub unsafe fn cstr_to_str_unchecked(vk_str: &[c_char]) -> &str {
    CStr::from_ptr(vk_str.as_ptr()).to_str().unwrap()
}

#[inline]
pub fn cstr_to_str(vk_str: &[c_char]) -> Result<&str> {
    let nul_pos = vk_str.iter().position(|&c| c == 0);
    let valid_slice = nul_pos.map(|pos| &vk_str[..=pos]);

    match valid_slice {
        Some(s) => unsafe { Ok(CStr::from_bytes_with_nul(mem::transmute(s))?.to_str()?) },
        None => Err(anyhow!("Invalid UTF-8 sequence")),
    }
}

#[test]
pub fn test_hello_world() -> Result<()> {
    let device: WrappedDeviceRef = WrappedDevice::new(true, &VALIDATION_LAYERS, "atodium_optics", 0, "atodium_optics_test", 0, API_VERSION, &DEVICE_EXTENSIONS)?.into();
    let buffer_allocator: RenderBufferAllocatorRef = RenderBufferAllocator::new(device.clone())?.into();

    let include_structure = shader_compiler::load_shaders();
    let pipeline_desc = PipelineDesc::default().compute_path("render.comp.glsl".into());
    let pipeline = WrappedPipeline::new(device.clone(), &buffer_allocator, pipeline_desc, &include_structure, None)?;

    let buffer = buffer_allocator.allocate(800 * 600 * 3 * 4, BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuToCpu)?;

    let descriptor = WrappedDescriptorSet::new(device.clone(), pipeline.descriptor_set_layouts[0], pipeline.reflection.binding_map.clone())?;
    descriptor.write_storage_buffer(DescriptorId::Index(0), &buffer)?;

    let render_width = 800;
    let render_height = 600;
    let workgroup_width = 16;
    let workgroup_height = 8;

    device.single_time_command(|device, cmd_buf| unsafe {
        pipeline.bind(cmd_buf);
        descriptor.bind(cmd_buf, &pipeline);

        device.cmd_dispatch(cmd_buf,
                            (render_width + workgroup_width - 1) / workgroup_width,
                            (render_height + workgroup_height - 1) / workgroup_height,
                            1);
    })?;

    let image_data: Vec<f32> = buffer_allocator.download_data(&buffer)?;

    let image = ImageBuffer::from_fn(render_width, render_height, |x, y| {
        let idx = ((y * render_width + x) * 3) as usize;
        image::Rgba([
            (image_data[idx] * 255.0) as u8,
            (image_data[idx + 1] * 255.0) as u8,
            (image_data[idx + 2] * 255.0) as u8,
            255,
        ])
    });

    image.save_with_format(lib_root().join("output").join("hello_world.png"), ImageFormat::Png)?;

    Ok(())
}
