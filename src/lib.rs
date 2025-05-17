use crate::render_resource::render_image::ImageDesc;
use crate::rt::{blas, tlas};
use crate::util::OutputFormat;
use crate::vk_context::descriptor_set::{DescriptorId, WrappedDescriptorSet};
use crate::vk_context::pipeline::{PipelineDesc, WrappedPipeline};
use anyhow::Result;
use ash::vk;
use ash::vk::{AccessFlags, BufferUsageFlags, DependencyFlags, DeviceSize, Format, ImageLayout, ImageTiling, ImageUsageFlags, MemoryBarrier, MemoryPropertyFlags, PipelineStageFlags};
use glam::Vec4;
use gpu_allocator::MemoryLocation;
use image::{ImageBuffer, ImageFormat};
use log::{error, info};
use std::slice;

pub mod model;
pub mod render_resource;
pub mod rt;
pub mod util;
pub mod vk_context;

pub fn test_hello_world() -> Result<()> {
    let (device, buffer_allocator, _, include_structure) = vk_context::init_vulkan_context(true, "test_hello_world", vk::make_api_version(0, 1, 1, 1))?;

    let pipeline_desc = PipelineDesc::default().compute_path("test/hello_world.comp.glsl".into());
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

    image.save_with_format(util::lib_root().join("output").join("hello_world.hdr"), ImageFormat::Hdr)?;

    Ok(())
}

pub fn test_cornell() -> Result<()> {
    let (device, allocator, image_allocator, include_structure) = vk_context::init_vulkan_context(true, "test_hello_world", vk::make_api_version(0, 1, 1, 1))?;

    let model = model::load_gltf(device.clone(), &allocator, &image_allocator, util::lib_root().join("models/cornell.gltf").to_str().unwrap())?;

    info!("Render model loaded");

    let vertices = model
        .meshes
        .iter()
        .map(|(mesh, transform)| mesh.mesh_buffer.vertices.iter().map(|&vertex| (*transform) * vertex.pos).collect::<Vec<_>>())
        .flatten()
        .collect::<Vec<_>>();

    let vertices_buffer = allocator.allocate(
        (vertices.len() * size_of::<Vec4>()) as DeviceSize,
        BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        MemoryLocation::GpuOnly,
    )?;

    allocator.upload_data(&vertices_buffer, &vertices)?;

    let blas = model
        .meshes
        .iter()
        .filter_map(|(mesh, _)| match blas::create_blas(device.clone(), &allocator, &mesh.mesh_buffer) {
            Ok(blas) => Some(blas),
            Err(error) => {
                error!("{:?}", error);
                None
            }
        })
        .collect::<Vec<_>>();

    info!("Bottom-level acceleration structures created");

    let tlas = tlas::create_tlas(device.clone(), &allocator, &blas, slice::from_ref(&model))?;

    info!("Top-level acceleration structure created");

    let pipeline_desc = PipelineDesc::default().compute_path("test/render.comp.glsl".into());
    let pipeline = WrappedPipeline::new(device.clone(), &allocator, pipeline_desc, &include_structure, None)?;

    let buffer = allocator.allocate(800 * 600 * 4 * 4, BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuToCpu)?;

    let descriptor = WrappedDescriptorSet::new(device.clone(), &pipeline, 0)?;
    descriptor.write_storage_buffer(DescriptorId::Index(0), &buffer)?;
    descriptor.write_acceleration_structure(DescriptorId::Index(1), tlas.handle)?;
    descriptor.write_storage_buffer(DescriptorId::Index(2), &vertices_buffer)?;
    descriptor.write_storage_buffer(DescriptorId::Index(3), &(model.meshes[0].0.mesh_buffer.index_buffer))?;

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

    let pixels: Vec<[f32; 4]> = allocator.download_data(&buffer)?;

    util::output_image(&util::lib_root().join("output").join("cornell.hdr"), render_width, render_height, &pixels, OutputFormat::Hdr)?;

    Ok(())
}

pub fn test_rt_pipeline() -> Result<()> {
    let (device, allocator, image_allocator, include_structure) = vk_context::init_vulkan_context(true, "test_hello_world", vk::make_api_version(0, 1, 1, 1))?;

    let model = model::load_gltf(device.clone(), &allocator, &image_allocator, util::lib_root().join("models/cornell.gltf").to_str().unwrap())?;

    info!("Render model loaded");

    let vertices = model
        .meshes
        .iter()
        .map(|(mesh, transform)| mesh.mesh_buffer.vertices.iter().map(|&vertex| (*transform) * vertex.pos).collect::<Vec<_>>())
        .flatten()
        .collect::<Vec<_>>();

    let vertices_buffer = allocator.allocate(
        (vertices.len() * size_of::<Vec4>()) as DeviceSize,
        BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        MemoryLocation::GpuOnly,
    )?;

    allocator.upload_data(&vertices_buffer, &vertices)?;

    let blas = model
        .meshes
        .iter()
        .filter_map(|(mesh, _)| match blas::create_blas(device.clone(), &allocator, &mesh.mesh_buffer) {
            Ok(blas) => Some(blas),
            Err(error) => {
                error!("{:?}", error);
                None
            }
        })
        .collect::<Vec<_>>();

    info!("Bottom-level acceleration structures created");

    let tlas = tlas::create_tlas(device.clone(), &allocator, &blas, slice::from_ref(&model))?;

    info!("Top-level acceleration structure created");

    let pipeline_desc = PipelineDesc::default()
        .raygen_path("test/rt.rgen.glsl".into())
        .hit_path("test/rt.rchit.glsl".into())
        .miss_path("test/rt.rmiss.glsl".into());
    let pipeline = WrappedPipeline::new(device.clone(), &allocator, pipeline_desc, &include_structure, None)?;

    let render_width = 800;
    let render_height = 600;

    let mut shader_image = image_allocator.allocate(
        ImageDesc::default_2d(render_width, render_height, Format::R32G32B32A32_SFLOAT, ImageUsageFlags::STORAGE | ImageUsageFlags::TRANSFER_SRC),
        MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let mut host_image = image_allocator.allocate(
        ImageDesc::default_2d(
            render_width,
            render_height,
            Format::R32G32B32A32_SFLOAT,
            ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST,
        )
        .tiling(ImageTiling::LINEAR),
        MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT | MemoryPropertyFlags::HOST_CACHED,
    )?;

    image_allocator.transition_layout(&mut shader_image, ImageLayout::GENERAL)?;
    image_allocator.transition_layout(&mut host_image, ImageLayout::TRANSFER_DST_OPTIMAL)?;

    let descriptor = WrappedDescriptorSet::new(device.clone(), &pipeline, 0)?;
    descriptor.write_acceleration_structure(DescriptorId::Index(0), tlas.handle)?;
    descriptor.write_storage_image(DescriptorId::Index(1), &shader_image)?;
    descriptor.write_storage_buffer(DescriptorId::Index(2), &vertices_buffer)?;
    descriptor.write_storage_buffer(DescriptorId::Index(3), &(model.meshes[0].0.mesh_buffer.index_buffer))?;

    device.single_time_command(|cmd_buf| unsafe {
        pipeline.bind(cmd_buf);
        descriptor.bind(cmd_buf, &pipeline);

        let sbt = pipeline.raytracing_sbt.as_ref().unwrap();

        device.rt_pipeline_device.cmd_trace_rays(
            cmd_buf,
            &sbt.raygen_region,
            &sbt.miss_region,
            &sbt.closest_hit_region,
            &sbt.callable_region,
            render_width,
            render_height,
            1,
        );
    })?;

    info!("Ray tracing rendering finished");

    image_allocator.copy_image(&shader_image, &host_image, None)?;

    let pixels = image_allocator.acquire_pixels(&mut host_image, None)?;

    util::output_image(&util::lib_root().join("output").join("cornell_pipelined.hdr"), render_width, render_height, &pixels, OutputFormat::Hdr)?;

    Ok(())
}
