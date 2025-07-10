use crate::memory::render_image::ImageDesc;
use crate::render::descriptor_set::{DescriptorId, WrappedDescriptorSet};
use crate::render::pipeline::{PipelineDesc, WrappedPipeline};
use crate::render::shader_builder;
use crate::rt::{blas, tlas};
use crate::util::OutputFormat;
use anyhow::Result;
use ash::vk;
use ash::vk::{AccessFlags, BufferUsageFlags, DependencyFlags, DeviceSize, Format, ImageLayout, ImageTiling, ImageUsageFlags, MemoryBarrier, MemoryPropertyFlags, PipelineStageFlags};
use glam::Vec4;
use gpu_allocator::MemoryLocation;
use image::codecs::hdr::HdrEncoder;
use log::{error, info};
use std::fs::File;
use std::{mem, slice};

pub mod memory;
pub mod model;
pub mod render;
pub mod rt;
pub mod util;

pub fn test_hello_world() -> Result<()> {
    let (device, buffer_allocator, _, _) = render::init_vulkan_context(true, "test_hello_world", vk::make_api_version(0, 1, 1, 1))?;

    let shaders = shader_builder::compile_spirv_shaders();

    let pipeline_desc = PipelineDesc::default().compute_name("test::hello_world::main_cs".into());
    let pipeline = WrappedPipeline::new(device.clone(), &buffer_allocator, pipeline_desc, &shaders, None)?;

    let buffer = buffer_allocator.allocate(800 * 600 * 4 * 4, BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuToCpu)?;

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

    let pixels: Vec<[f32; 4]> = buffer_allocator.download_data(&buffer)?;

    let pixels = pixels.iter().map(|pixel| image::Rgb([pixel[0], pixel[1], pixel[2]])).collect::<Vec<_>>();
    let mut file = File::create(util::lib_root().join("output").join("hello_world.hdr"))?;
    let encoder = HdrEncoder::new(&mut file);
    encoder.encode(&pixels, render_width as usize, render_height as usize)?;

    Ok(())
}

pub fn test_cornell() -> Result<()> {
    let (device, allocator, image_allocator, _) = render::init_vulkan_context(true, "test_hello_world", vk::make_api_version(0, 1, 1, 1))?;

    let shaders = shader_builder::compile_spirv_shaders();

    let model = model::load_gltf(device.clone(), &allocator, &image_allocator, util::lib_root().join("models/cornell.gltf").to_str().unwrap())?;

    info!("Render model loaded");

    let vertices = model
        .meshes
        .iter()
        .map(|(mesh, transform)| mesh.mesh_buffer.vertices.iter().map(|&vertex| (*transform) * vertex.pos).collect::<Vec<_>>())
        .flatten()
        .collect::<Vec<_>>();

    let vertices_buffer = allocator.allocate(
        (vertices.len() * mem::size_of::<Vec4>()) as DeviceSize,
        BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        MemoryLocation::GpuOnly,
    )?;

    allocator.upload_data(&vertices_buffer, &vertices)?;

    let blas = model
        .meshes
        .iter()
        .enumerate()
        .filter_map(|(i, (mesh, _))| match blas::create_blas(device.clone(), &allocator, &mesh.mesh_buffer, i as _) {
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

    let pipeline_desc = PipelineDesc::default().compute_name("test::cornell::main_cs".into());
    let pipeline = WrappedPipeline::new(device.clone(), &allocator, pipeline_desc, &shaders, None)?;

    let buffer = allocator.allocate(800 * 600 * 4 * 4, BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuToCpu)?;

    let descriptor = WrappedDescriptorSet::new(device.clone(), &pipeline, 0)?;
    descriptor.write_storage_buffer(DescriptorId::Index(0), &buffer)?;
    descriptor.write_tlas(DescriptorId::Index(1), &tlas)?;
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

    util::output_image(&util::lib_root().join("output").join("cornell.png"), render_width, render_height, &pixels, OutputFormat::Png)?;

    Ok(())
}

pub fn test_rt_pipeline() -> Result<()> {
    let (device, allocator, image_allocator, _) = render::init_vulkan_context(true, "test_hello_world", vk::make_api_version(0, 1, 1, 1))?;
    let shaders = shader_builder::compile_spirv_shaders();

    let model = model::load_gltf(device.clone(), &allocator, &image_allocator, util::lib_root().join("models/cornell_color.gltf").to_str().unwrap())?;

    info!("Render model loaded");

    let instance_metadata_buffer = model.write_instance_metadata_to_buffer(&allocator)?;
    let vertices_buffer = model.write_vertices_to_buffer(&allocator)?;
    let indices_buffer = model.write_indices_to_buffer(&allocator)?;
    let materials_buffer = model.write_material_to_buffer(&allocator)?;

    let blas = model.build_blas(device.clone(), &allocator);

    info!("Bottom-level acceleration structures created");

    let tlas = tlas::create_tlas(device.clone(), &allocator, &blas, slice::from_ref(&model))?;

    info!("Top-level acceleration structure created");

    let pipeline_desc = PipelineDesc::default()
        .raygen_name("test::rt_pipeline::main_rgen".into())
        .hit_name("test::rt_pipeline::main_rchit".into())
        .miss_name("test::rt_pipeline::main_rmiss".into());
    let pipeline = WrappedPipeline::new(device.clone(), &allocator, pipeline_desc, &shaders, None)?;

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
    descriptor.write_tlas(DescriptorId::Index(0), &tlas)?;
    descriptor.write_storage_image(DescriptorId::Index(1), &shader_image)?;
    descriptor.write_storage_buffer(DescriptorId::Index(2), &vertices_buffer)?;
    descriptor.write_storage_buffer(DescriptorId::Index(3), &indices_buffer)?;
    descriptor.write_storage_buffer(DescriptorId::Index(4), &instance_metadata_buffer)?;
    descriptor.write_storage_buffer(DescriptorId::Index(5), &materials_buffer)?;

    device.single_time_command(|cmd_buf| {
        pipeline.bind(cmd_buf);
        descriptor.bind(cmd_buf, &pipeline);
        device.cmd_trace_rays(cmd_buf, pipeline.raytracing_sbt.as_ref().unwrap(), shader_image.extent());
    })?;

    info!("Ray tracing rendering finished");

    image_allocator.copy_image(&shader_image, &host_image, None)?;

    let pixels = image_allocator.acquire_pixels(&mut host_image, None)?;

    util::output_image(&util::lib_root().join("output").join("cornell_pipelined.hdr"), render_width, render_height, &pixels, OutputFormat::Hdr)?;

    Ok(())
}
