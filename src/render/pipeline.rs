use crate::memory::render_buffer::{RenderBuffer, RenderBufferAllocator};
use crate::render;
use crate::render::device::{WrappedDevice, WrappedDeviceRef};
use crate::render::glsl_shader_compiler;
use crate::render::shader_builder::SpirvShaders;
use crate::render::shader_reflection::ShaderReflection;
use anyhow::{anyhow, bail, Result};
use ash::vk;
use ash::vk::{
    BlendFactor, BlendOp, BufferUsageFlags, ColorComponentFlags, CommandBuffer, CompareOp, ComputePipelineCreateInfo, DeferredOperationKHR, DescriptorSetLayout, DeviceSize, DynamicState, Format,
    FrontFace, GraphicsPipelineCreateInfo, LogicOp, Pipeline, PipelineBindPoint, PipelineCache, PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateInfo,
    PipelineDepthStencilStateCreateInfo, PipelineDynamicStateCreateInfo, PipelineInputAssemblyStateCreateInfo, PipelineLayout, PipelineMultisampleStateCreateInfo,
    PipelineRasterizationStateCreateInfo, PipelineRenderingCreateInfo, PipelineShaderStageCreateInfo, PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateInfo, PolygonMode,
    PrimitiveTopology, RayTracingPipelineCreateInfoKHR, RayTracingShaderGroupCreateInfoKHR, RayTracingShaderGroupTypeKHR, RenderPass, SampleCountFlags, ShaderModule, ShaderModuleCreateInfo,
    ShaderStageFlags, StencilOp, StencilOpState, StridedDeviceAddressRegionKHR, VertexInputAttributeDescription, VertexInputBindingDescription,
};
use gpu_allocator::MemoryLocation;
use std::ffi::CString;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::slice;

#[derive(Clone)]
pub struct PipelineDesc {
    pub vertex_name: Option<String>,
    pub fragment_name: Option<String>,
    pub compute_name: Option<String>,
    pub raygen_name: Option<String>,
    pub miss_name: Option<String>,
    pub closest_hit_name: Option<String>,

    pub vertex_input_binding_descriptions: Vec<VertexInputBindingDescription>,
    pub vertex_input_attribute_descriptions: Vec<VertexInputAttributeDescription>,
    pub color_attachment_formats: Vec<Format>,
    pub depth_stencil_attachment_format: Format,
}

pub struct WrappedPipeline {
    device: WrappedDeviceRef,

    pub handle: Pipeline,
    pub pipeline_layout: PipelineLayout,
    pub descriptor_set_layouts: Vec<DescriptorSetLayout>,
    pub shader_modules: Vec<ShaderModule>,
    pub reflection: ShaderReflection,
    pub pipeline_desc: PipelineDesc,
    pub pipeline_type: PipelineType,
    pub raytracing_sbt: Option<RayTracingSbt>,
}

impl Deref for WrappedPipeline {
    type Target = Pipeline;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl Drop for WrappedPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device.destroy_pipeline(self.handle, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.descriptor_set_layouts
                .iter()
                .for_each(|&descriptor_set_layout| self.device.destroy_descriptor_set_layout(descriptor_set_layout, None));
            self.shader_modules.iter().for_each(|&shader_module| self.device.destroy_shader_module(shader_module, None));
        }
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum PipelineType {
    Graphics,
    Compute,
    Raytracing,
}

#[allow(dead_code)]
pub struct RayTracingSbt {
    pub sbt_buffer: RenderBuffer,
    pub raygen_region: StridedDeviceAddressRegionKHR,
    pub miss_region: StridedDeviceAddressRegionKHR,
    pub closest_hit_region: StridedDeviceAddressRegionKHR,
    pub callable_region: StridedDeviceAddressRegionKHR,
}

impl Default for PipelineDesc {
    fn default() -> Self {
        Self {
            vertex_name: None,
            fragment_name: None,
            compute_name: None,
            raygen_name: None,
            miss_name: None,
            closest_hit_name: None,
            vertex_input_binding_descriptions: Vec::new(),
            vertex_input_attribute_descriptions: Vec::new(),
            color_attachment_formats: Vec::new(),
            depth_stencil_attachment_format: Format::UNDEFINED,
        }
    }
}

impl PipelineDesc {
    pub fn is_graphics_pipeline(&self) -> bool {
        self.vertex_name.is_some() && self.fragment_name.is_some()
    }

    pub fn is_compute_pipeline(&self) -> bool {
        self.compute_name.is_some()
    }

    pub fn is_raytracing_pipeline(&self) -> bool {
        self.raygen_name.is_some() && self.miss_name.is_some() && self.closest_hit_name.is_some()
    }

    pub fn vertex_name(mut self, name: String) -> Self {
        self.vertex_name = Some(name);
        self
    }

    pub fn fragment_name(mut self, name: String) -> Self {
        self.fragment_name = Some(name);
        self
    }

    pub fn compute_name(mut self, name: String) -> Self {
        self.compute_name = Some(name);
        self
    }

    pub fn raygen_name(mut self, name: String) -> Self {
        self.raygen_name = Some(name);
        self
    }

    pub fn miss_name(mut self, name: String) -> Self {
        self.miss_name = Some(name);
        self
    }

    pub fn hit_name(mut self, name: String) -> Self {
        self.closest_hit_name = Some(name);
        self
    }

    pub fn vertex_input_binding_descriptions(mut self, descriptions: Vec<VertexInputBindingDescription>) -> Self {
        self.vertex_input_binding_descriptions = descriptions;
        self
    }

    pub fn vertex_input_attribute_descriptions(mut self, descriptions: Vec<VertexInputAttributeDescription>) -> Self {
        self.vertex_input_attribute_descriptions = descriptions;
        self
    }

    pub fn color_attachment_formats(mut self, formats: Vec<Format>) -> Self {
        self.color_attachment_formats = formats;
        self
    }

    pub fn depth_stencil_attachment_format(mut self, format: Format) -> Self {
        self.depth_stencil_attachment_format = format;
        self
    }
}

impl Hash for PipelineDesc {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vertex_name.hash(state);
        self.fragment_name.hash(state);
        self.compute_name.hash(state);
        self.raygen_name.hash(state);
        self.miss_name.hash(state);
        self.closest_hit_name.hash(state);
    }
}

impl PartialEq for PipelineDesc {
    fn eq(&self, other: &Self) -> bool {
        self.vertex_name == other.vertex_name
            && self.fragment_name == other.fragment_name
            && self.compute_name == other.compute_name
            && self.raygen_name == other.raygen_name
            && self.miss_name == other.miss_name
            && self.closest_hit_name == other.closest_hit_name
    }
}

impl WrappedPipeline {
    pub fn new(
        device: WrappedDeviceRef,
        buffer_allocator: &RenderBufferAllocator,
        pipeline_desc: PipelineDesc,
        shaders: &SpirvShaders,
        bindless_descriptor_set_layout: Option<DescriptorSetLayout>,
    ) -> Result<WrappedPipeline> {
        let pipeline_type = if pipeline_desc.is_graphics_pipeline() {
            PipelineType::Graphics
        } else if pipeline_desc.is_compute_pipeline() {
            PipelineType::Compute
        } else if pipeline_desc.is_raytracing_pipeline() {
            PipelineType::Raytracing
        } else {
            bail!("Pipeline description is incomplete");
        };

        let (reflection, pipeline_layout, descriptor_set_layouts, shader_modules) = match pipeline_type {
            PipelineType::Graphics => create_graphics_shader_modules(
                &device,
                &pipeline_desc.vertex_name.as_ref().unwrap(),
                &pipeline_desc.fragment_name.as_ref().unwrap(),
                shaders,
                bindless_descriptor_set_layout,
            ),
            PipelineType::Compute => create_compute_shader_modules(&device, &pipeline_desc.compute_name.as_ref().unwrap(), shaders, bindless_descriptor_set_layout),
            PipelineType::Raytracing => create_raytracing_shader_modules(
                &device,
                &pipeline_desc.raygen_name.as_ref().unwrap(),
                &pipeline_desc.miss_name.as_ref().unwrap(),
                &pipeline_desc.closest_hit_name.as_ref().unwrap(),
                shaders,
                bindless_descriptor_set_layout,
            ),
        }?;

        let handle = match pipeline_type {
            PipelineType::Graphics => create_graphics_pipeline(
                &device,
                &shader_modules,
                &pipeline_desc.color_attachment_formats,
                pipeline_desc.depth_stencil_attachment_format,
                pipeline_layout,
                &pipeline_desc,
            ),
            PipelineType::Compute => create_compute_pipeline(&device, &shader_modules, pipeline_layout, &pipeline_desc),
            PipelineType::Raytracing => create_raytracing_pipeline(&device, &shader_modules, pipeline_layout, &pipeline_desc),
        }?;

        let raytracing_sbt = if pipeline_type == PipelineType::Raytracing {
            Some(create_raytracing_sbt(&device, buffer_allocator, handle, 1)?)
        } else {
            None
        };

        let pipeline = WrappedPipeline {
            device,
            handle,
            pipeline_layout,
            descriptor_set_layouts,
            shader_modules,
            reflection,
            pipeline_desc,
            pipeline_type,
            raytracing_sbt,
        };

        Ok(pipeline)
    }

    #[inline]
    pub fn bind_point(&self) -> PipelineBindPoint {
        match self.pipeline_type {
            PipelineType::Graphics => PipelineBindPoint::GRAPHICS,
            PipelineType::Compute => PipelineBindPoint::COMPUTE,
            PipelineType::Raytracing => PipelineBindPoint::RAY_TRACING_KHR,
        }
    }

    pub fn bind(&self, cmd_buf: CommandBuffer) {
        unsafe { self.device.cmd_bind_pipeline(cmd_buf, self.bind_point(), self.handle) };
    }
}

fn create_graphics_shader_modules(
    device: &WrappedDevice,
    vertex_shader_name: &str,
    fragment_shader_name: &str,
    shaders: &SpirvShaders,
    bindless_descriptor_set_layout: Option<DescriptorSetLayout>,
) -> Result<(ShaderReflection, PipelineLayout, Vec<DescriptorSetLayout>, Vec<ShaderModule>)> {
    let vertex_shader = shaders.get(vertex_shader_name).ok_or_else(|| anyhow!("Vertex shader [ {} ] not found", vertex_shader_name))?;
    let fragment_shader = shaders.get(fragment_shader_name).ok_or_else(|| anyhow!("Fragment shader [ {} ] not found", fragment_shader_name))?;

    let reflection = ShaderReflection::new(&[vertex_shader.as_binary_u8(), fragment_shader.as_binary_u8()])?;

    let (pipeline_layout, descriptor_set_layouts, _) = glsl_shader_compiler::create_pipeline_layout(device, &reflection, bindless_descriptor_set_layout);

    let vertex_shader_module = create_shader_module(device, vertex_shader.as_binary())?;
    let fragment_shader_module = create_shader_module(device, fragment_shader.as_binary())?;

    let shader_modules = vec![vertex_shader_module, fragment_shader_module];

    Ok((reflection, pipeline_layout, descriptor_set_layouts, shader_modules))
}

fn create_graphics_pipeline(
    device: &WrappedDevice,
    shader_modules: &[ShaderModule],
    color_attachment_formats: &[Format],
    depth_stencil_attachment_format: Format,
    pipeline_layout: PipelineLayout,
    pipeline_desc: &PipelineDesc,
) -> Result<Pipeline> {
    let vertex_entry_name = CString::new(pipeline_desc.vertex_name.as_ref().unwrap().as_str())?;
    let fragment_entry_name = CString::new(pipeline_desc.fragment_name.as_ref().unwrap().as_str())?;

    let shader_stage_create_infos = vec![
        PipelineShaderStageCreateInfo {
            module: shader_modules[0],
            p_name: vertex_entry_name.as_ptr(),
            stage: ShaderStageFlags::VERTEX,
            ..Default::default()
        },
        PipelineShaderStageCreateInfo {
            module: shader_modules[1],
            p_name: fragment_entry_name.as_ptr(),
            stage: ShaderStageFlags::FRAGMENT,
            ..Default::default()
        },
    ];

    let vertex_input_state_info = PipelineVertexInputStateCreateInfo::default()
        .vertex_attribute_descriptions(pipeline_desc.vertex_input_attribute_descriptions.as_slice())
        .vertex_binding_descriptions(pipeline_desc.vertex_input_binding_descriptions.as_slice());

    let vertex_input_assembly_state_info = PipelineInputAssemblyStateCreateInfo::default().topology(PrimitiveTopology::TRIANGLE_LIST);

    let viewport_state_info = PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);

    let rasterization_info = PipelineRasterizationStateCreateInfo::default()
        .front_face(FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0)
        .polygon_mode(PolygonMode::FILL);

    let multisample_state_info = PipelineMultisampleStateCreateInfo::default().rasterization_samples(SampleCountFlags::TYPE_1);

    let stencil_op_state = StencilOpState::default()
        .fail_op(StencilOp::KEEP)
        .pass_op(StencilOp::KEEP)
        .depth_fail_op(StencilOp::KEEP)
        .compare_op(CompareOp::ALWAYS);

    let depth_stencil_state_info = PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(CompareOp::LESS_OR_EQUAL)
        .front(stencil_op_state)
        .back(stencil_op_state)
        .max_depth_bounds(1.0);

    let color_blend_attachment_states = vec![
        PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .src_color_blend_factor(BlendFactor::SRC_COLOR)
            .dst_color_blend_factor(BlendFactor::ONE_MINUS_DST_COLOR)
            .color_blend_op(BlendOp::ADD)
            .src_alpha_blend_factor(BlendFactor::ZERO)
            .dst_alpha_blend_factor(BlendFactor::ZERO)
            .alpha_blend_op(BlendOp::ADD)
            .color_write_mask(ColorComponentFlags::R | ColorComponentFlags::G | ColorComponentFlags::B | ColorComponentFlags::A);
        color_attachment_formats.len()
    ];

    let color_blend_state = PipelineColorBlendStateCreateInfo::default().logic_op(LogicOp::CLEAR).attachments(&color_blend_attachment_states);

    let dynamic_state = [DynamicState::VIEWPORT, DynamicState::SCISSOR];

    let dynamic_state_info = PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_state);

    let mut rendering_info = PipelineRenderingCreateInfo::default()
        .color_attachment_formats(color_attachment_formats)
        .depth_attachment_format(depth_stencil_attachment_format)
        .stencil_attachment_format(Format::UNDEFINED);

    let graphic_pipeline_info = GraphicsPipelineCreateInfo::default()
        .stages(&shader_stage_create_infos)
        .vertex_input_state(&vertex_input_state_info)
        .input_assembly_state(&vertex_input_assembly_state_info)
        .viewport_state(&viewport_state_info)
        .rasterization_state(&rasterization_info)
        .multisample_state(&multisample_state_info)
        .depth_stencil_state(&depth_stencil_state_info)
        .color_blend_state(&color_blend_state)
        .dynamic_state(&dynamic_state_info)
        .layout(pipeline_layout)
        .render_pass(RenderPass::null())
        .push_next(&mut rendering_info);

    match unsafe { device.create_graphics_pipelines(PipelineCache::null(), slice::from_ref(&graphic_pipeline_info), None) } {
        Ok(graphics_pipelines) => Ok(graphics_pipelines[0]),
        Err((_, result)) => Err(anyhow!(result)),
    }
}

fn create_compute_shader_modules(
    device: &WrappedDevice,
    compute_shader_name: &str,
    shaders: &SpirvShaders,
    bindless_descriptor_set_layout: Option<DescriptorSetLayout>,
) -> Result<(ShaderReflection, PipelineLayout, Vec<DescriptorSetLayout>, Vec<ShaderModule>)> {
    let compute_shader = shaders.get(compute_shader_name).ok_or_else(|| anyhow!("Compute shader [ {} ] not found", compute_shader_name))?;

    let reflection = ShaderReflection::new(&[compute_shader.as_binary_u8()])?;

    let (pipeline_layout, descriptor_set_layouts, _) = glsl_shader_compiler::create_pipeline_layout(device, &reflection, bindless_descriptor_set_layout);

    let compute_shader_module = create_shader_module(device, compute_shader.as_binary())?;

    let shader_modules = vec![compute_shader_module];

    Ok((reflection, pipeline_layout, descriptor_set_layouts, shader_modules))
}

fn create_compute_pipeline(device: &WrappedDevice, shader_modules: &[ShaderModule], pipeline_layout: PipelineLayout, pipeline_desc: &PipelineDesc) -> Result<Pipeline> {
    let compute_entry_cstring = CString::new(pipeline_desc.compute_name.as_ref().unwrap().as_str())?;

    let shader_stage_create_infos = vec![PipelineShaderStageCreateInfo {
        module: shader_modules[0],
        p_name: compute_entry_cstring.as_ptr(),
        stage: ShaderStageFlags::COMPUTE,
        ..Default::default()
    }];

    let compute_pipeline_info = ComputePipelineCreateInfo::default().stage(shader_stage_create_infos[0]).layout(pipeline_layout);

    match unsafe { device.create_compute_pipelines(PipelineCache::null(), slice::from_ref(&compute_pipeline_info), None) } {
        Ok(compute_pipeline) => Ok(compute_pipeline[0]),
        Err((_, result)) => Err(anyhow!(result)),
    }
}

fn create_raytracing_shader_modules(
    device: &WrappedDevice,
    raygen_shader_name: &str,
    miss_shader_name: &str,
    closest_hit_shader_name: &str,
    shaders: &SpirvShaders,
    bindless_descriptor_set_layout: Option<DescriptorSetLayout>,
) -> Result<(ShaderReflection, PipelineLayout, Vec<DescriptorSetLayout>, Vec<ShaderModule>)> {
    let raygen_shader = shaders.get(raygen_shader_name).ok_or_else(|| anyhow!("Ray generation shader [ {} ] not found", raygen_shader_name))?;
    let miss_shader = shaders.get(miss_shader_name).ok_or_else(|| anyhow!("Miss shader [ {} ] not found", miss_shader_name))?;
    let closest_hit_shader = shaders
        .get(closest_hit_shader_name)
        .ok_or_else(|| anyhow!("Closest hit generation shader [ {} ] not found", closest_hit_shader_name))?;

    let reflection = ShaderReflection::new(&[raygen_shader.as_binary_u8(), miss_shader.as_binary_u8(), closest_hit_shader.as_binary_u8()])?;
    let (pipeline_layout, descriptor_set_layouts, _) = glsl_shader_compiler::create_pipeline_layout(device, &reflection, bindless_descriptor_set_layout);

    let raygen_shader_module = create_shader_module(device, raygen_shader.as_binary())?;
    let miss_shader_module = create_shader_module(device, miss_shader.as_binary())?;
    let closest_hit_shader_module = create_shader_module(device, closest_hit_shader.as_binary())?;

    let shader_modules = vec![raygen_shader_module, miss_shader_module, closest_hit_shader_module];

    Ok((reflection, pipeline_layout, descriptor_set_layouts, shader_modules))
}

fn create_raytracing_pipeline(device: &WrappedDevice, shader_modules: &[ShaderModule], pipeline_layout: PipelineLayout, pipeline_desc: &PipelineDesc) -> Result<Pipeline> {
    let raygen_entry_name = CString::new(pipeline_desc.raygen_name.as_ref().unwrap().as_str())?;
    let miss_entry_name = CString::new(pipeline_desc.miss_name.as_ref().unwrap().as_str())?;
    let closest_hit_shader_name = CString::new(pipeline_desc.closest_hit_name.as_ref().unwrap().as_str())?;

    let shader_stage_create_infos = vec![
        PipelineShaderStageCreateInfo {
            module: shader_modules[0],
            p_name: raygen_entry_name.as_ptr(),
            stage: ShaderStageFlags::RAYGEN_KHR,
            ..Default::default()
        },
        PipelineShaderStageCreateInfo {
            module: shader_modules[1],
            p_name: miss_entry_name.as_ptr(),
            stage: ShaderStageFlags::MISS_KHR,
            ..Default::default()
        },
        PipelineShaderStageCreateInfo {
            module: shader_modules[2],
            p_name: closest_hit_shader_name.as_ptr(),
            stage: ShaderStageFlags::CLOSEST_HIT_KHR,
            ..Default::default()
        },
    ];

    let shader_group_create_infos = [
        RayTracingShaderGroupCreateInfoKHR::default()
            .ty(RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(0)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
        RayTracingShaderGroupCreateInfoKHR::default()
            .ty(RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(1)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
        RayTracingShaderGroupCreateInfoKHR::default()
            .ty(RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
            .general_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(2)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
    ];

    let pipeline_create_info = RayTracingPipelineCreateInfoKHR::default()
        .max_pipeline_ray_recursion_depth(1)
        .layout(pipeline_layout)
        .stages(&shader_stage_create_infos)
        .groups(&shader_group_create_infos);

    match unsafe {
        device
            .rt_pipeline_device
            .create_ray_tracing_pipelines(DeferredOperationKHR::null(), PipelineCache::null(), slice::from_ref(&pipeline_create_info), None)
    } {
        Ok(rt_pipeline) => Ok(rt_pipeline[0]),
        Err((_, result)) => Err(anyhow!(result)),
    }
}

fn create_raytracing_sbt(device: &WrappedDevice, buffer_allocator: &RenderBufferAllocator, pipeline: Pipeline, closest_hit_count: u32) -> Result<RayTracingSbt> {
    let handle_size = device.rt_pipeline_properties.shader_group_handle_size as DeviceSize;
    let handle_alignment = device.rt_pipeline_properties.shader_group_handle_alignment as DeviceSize;
    let base_alignment = device.rt_pipeline_properties.shader_group_base_alignment as DeviceSize;

    let handle_size_aligned = render::align_up(handle_size, handle_alignment);

    let raygen_size = render::align_up(handle_size_aligned, base_alignment);
    let miss_size = render::align_up(handle_size_aligned, base_alignment);
    let closest_hit_size = render::align_up((closest_hit_count as DeviceSize) * handle_size_aligned, base_alignment);

    let handle_count = 2 + closest_hit_count;
    let sbt_buffer_size = raygen_size + miss_size + closest_hit_size;

    let shader_group_handles = unsafe {
        device
            .rt_pipeline_device
            .get_ray_tracing_shader_group_handles(pipeline, 0, handle_count, (handle_count as usize) * (handle_size as usize))?
    };

    let sbt_buffer = buffer_allocator.allocate(
        sbt_buffer_size,
        BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::SHADER_DEVICE_ADDRESS | BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
        MemoryLocation::GpuOnly,
    )?;

    let mut shader_group_handles_aligned = vec![0_u8; sbt_buffer_size as usize];
    for i in 0..handle_size {
        shader_group_handles_aligned[i as usize] = shader_group_handles[i as usize];
    }
    for i in 0..handle_size {
        shader_group_handles_aligned[(raygen_size + i) as usize] = shader_group_handles[(handle_size + i) as usize]
    }
    for c in 0..(closest_hit_count as DeviceSize) {
        for i in 0..handle_size {
            shader_group_handles_aligned[(raygen_size + miss_size + c * handle_size_aligned + i) as usize] = shader_group_handles[((2 + c) * handle_size + i) as usize]
        }
    }

    buffer_allocator.upload_data(&sbt_buffer, &shader_group_handles_aligned)?;

    let raygen_region = StridedDeviceAddressRegionKHR::default()
        .device_address(sbt_buffer.device_addr().unwrap())
        .stride(raygen_size)
        .size(raygen_size);

    let miss_region = StridedDeviceAddressRegionKHR::default()
        .device_address(sbt_buffer.device_addr().unwrap() + raygen_size)
        .stride(handle_size_aligned)
        .size(miss_size);

    let closest_hit_region = StridedDeviceAddressRegionKHR::default()
        .device_address(sbt_buffer.device_addr().unwrap() + raygen_size + miss_size)
        .stride(handle_size_aligned)
        .size(closest_hit_size);

    let callable_region = StridedDeviceAddressRegionKHR::default();

    Ok(RayTracingSbt {
        sbt_buffer,
        raygen_region,
        miss_region,
        closest_hit_region,
        callable_region,
    })
}

pub fn create_shader_module(device: &WrappedDevice, shader_code: &[u32]) -> Result<ShaderModule> {
    let shader_info = ShaderModuleCreateInfo::default().code(shader_code);

    Ok(unsafe { device.create_shader_module(&shader_info, None) }?)
}
