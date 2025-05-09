use crate::rt::{blas, tlas};
use crate::vk_context::descriptor_set::{DescriptorId, WrappedDescriptorSet};
use crate::vk_context::pipeline::{PipelineDesc, WrappedPipeline};
use anyhow::{Result, anyhow};
use ash::vk;
use ash::vk::{AccessFlags, BufferUsageFlags, DependencyFlags, MemoryBarrier, PipelineStageFlags};
use glam::Vec4;
use gpu_allocator::MemoryLocation;
use image::{ImageBuffer, ImageFormat};
use log::{error, info};
use std::ffi::{CStr, c_char};
use std::path::Path;
use std::{mem, slice};

pub mod model;
pub mod render_resource;
pub mod rt;
pub mod vk_context;

#[inline]
pub fn lib_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
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

pub fn test_hello_world() -> Result<()> {
    let (device, buffer_allocator, _, include_structure) = vk_context::init_vulkan_context(true, "test_hello_world", vk::make_api_version(0, 1, 1, 1))?;

    let pipeline_desc = PipelineDesc::default().compute_path("hello_world.comp.glsl".into());
    let pipeline = WrappedPipeline::new(device.clone(), &buffer_allocator, pipeline_desc, &include_structure, None)?;

    let buffer = buffer_allocator.allocate(800 * 600 * 3 * 4, BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuToCpu)?;

    let descriptor = WrappedDescriptorSet::new(device.clone(), &pipeline, 0)?;
    descriptor.write_storage_buffer(DescriptorId::Index(0), &buffer)?;

    let render_width = 800;
    let render_height = 600;
    let workgroup_width = 16;
    let workgroup_height = 8;

    device.single_time_command(|cmd_buf| unsafe {
        pipeline.bind(cmd_buf);
        descriptor.bind(cmd_buf, &pipeline);

        device.cmd_dispatch(
            cmd_buf,
            (render_width + workgroup_width - 1) / workgroup_width,
            (render_height + workgroup_height - 1) / workgroup_height,
            1,
        );
    })?;

    let image_data: Vec<f32> = buffer_allocator.download_data(&buffer)?;

    let image = ImageBuffer::from_fn(render_width, render_height, |x, y| {
        let idx = ((y * render_width + x) * 3) as usize;
        image::Rgb([image_data[idx], image_data[idx + 1], image_data[idx + 2]])
    });

    image.save_with_format(lib_root().join("output").join("hello_world.hdr"), ImageFormat::Hdr)?;

    Ok(())
}

pub fn test_cornell_hit() -> Result<()> {
    let (device, buffer_allocator, image_manager, include_structure) = vk_context::init_vulkan_context(true, "test_hello_world", vk::make_api_version(0, 1, 1, 1))?;

    let model = model::load_gltf(device.clone(), &buffer_allocator, &image_manager, lib_root().join("models/cornell.gltf").to_str().unwrap())?;

    info!("Render model loaded");

    let blas = model
        .meshes
        .iter()
        .filter_map(|(mesh, _)| match blas::create_blas(device.clone(), &buffer_allocator, &mesh.mesh_buffer) {
            Ok(blas) => Some(blas),
            Err(error) => {
                error!("{:?}", error);
                None
            }
        })
        .collect::<Vec<_>>();

    info!("Bottom-level acceleration structures created");

    let tlas = tlas::create_tlas(device.clone(), &buffer_allocator, &blas, &[model])?;

    info!("Top-level acceleration structure created");

    let pipeline_desc = PipelineDesc::default().compute_path("render.comp.glsl".into());
    let pipeline = WrappedPipeline::new(device.clone(), &buffer_allocator, pipeline_desc, &include_structure, None)?;

    let buffer = buffer_allocator.allocate(800 * 600 * 4 * 4, BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuToCpu)?;

    let descriptor = WrappedDescriptorSet::new(device.clone(), &pipeline, 0)?;
    descriptor.write_storage_buffer(DescriptorId::Index(0), &buffer)?;
    descriptor.write_acceleration_structure(DescriptorId::Index(1), tlas.handle)?;

    let render_width = 800;
    let render_height = 600;
    let workgroup_width = 16;
    let workgroup_height = 8;

    device.single_time_command(|cmd_buf| unsafe {
        pipeline.bind(cmd_buf);
        descriptor.bind(cmd_buf, &pipeline);

        device.cmd_dispatch(
            cmd_buf,
            (render_width + workgroup_width - 1) / workgroup_width,
            (render_height + workgroup_height - 1) / workgroup_height,
            1,
        );

        let memory_barrier = MemoryBarrier::default().src_access_mask(AccessFlags::SHADER_WRITE).dst_access_mask(AccessFlags::HOST_READ);

        device.cmd_pipeline_barrier(
            cmd_buf,
            PipelineStageFlags::COMPUTE_SHADER,
            PipelineStageFlags::HOST,
            DependencyFlags::empty(),
            slice::from_ref(&memory_barrier),
            &[],
            &[],
        );
    })?;

    info!("Compute shader command finished");

    let image_data: Vec<Vec4> = buffer_allocator.download_data(&buffer)?;

    let image = ImageBuffer::from_fn(render_width, render_height, |x, y| {
        let idx = (y * render_width + x) as usize;

        let r = image_data[idx].x;
        let g = image_data[idx].y;
        let b = image_data[idx].z;

        image::Rgb([r, g, b])
    });

    image.save_with_format(lib_root().join("output").join("cornell.hdr"), ImageFormat::Hdr)?;

    Ok(())
}
