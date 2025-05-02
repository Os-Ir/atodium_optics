use crate::vk_context::descriptor_set;
use crate::vk_context::device::WrappedDevice;
use crate::vk_context::shader_reflection::ShaderReflection;
use anyhow::{Result, anyhow};
use ash::vk::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, PipelineLayout, PipelineLayoutCreateInfo, PushConstantRange, ShaderModule, ShaderModuleCreateInfo, ShaderStageFlags,
};
use lazy_static::lazy_static;
use log::{error, info};
use rspirv_reflect::BindingCount;
use shaderc::{CompilationArtifact, CompileOptions, EnvVersion, ResolvedInclude, ShaderKind, TargetEnv};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use walkdir::WalkDir;

pub struct ShaderIncludeStructure {
    pub shader_sources: HashMap<PathBuf, String>,
}

impl ShaderIncludeStructure {
    pub fn new(shader_sources: HashMap<PathBuf, String>) -> Self {
        Self { shader_sources }
    }

    pub fn get_shader_source(&self, path: &PathBuf) -> Option<String> {
        self.shader_sources.get(path).cloned()
    }
}

#[inline]
pub fn shader_base_dir() -> PathBuf {
    crate::lib_root().join("shaders")
}

#[inline]
pub fn shader_dir(shader_path: &str) -> PathBuf {
    shader_base_dir().join(shader_path)
}

pub fn load_shaders() -> ShaderIncludeStructure {
    let shader_sources = WalkDir::new(shader_base_dir())
        .into_iter()
        .filter_map(|entry| {
            let entry = entry.ok()?;

            if !entry.file_type().is_file() {
                return None;
            }

            info!("Loading shader: [ {} ]", entry.path().display());

            fs::read_to_string(entry.path()).ok().map(|content| (entry.into_path(), content))
        })
        .collect();

    ShaderIncludeStructure { shader_sources }
}

lazy_static! {
    static ref SHADER_COMPILER: shaderc::Compiler = shaderc::Compiler::new().unwrap();
}

pub fn compile_glsl_shader(shader_path: &str, shader_kind: ShaderKind, include_structure: &ShaderIncludeStructure) -> Result<CompilationArtifact> {
    let shader_path_buf = shader_dir(shader_path);

    let mut options = CompileOptions::new()?;

    options.add_macro_definition("EP", Some("main"));
    options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_2 as u32);
    options.set_generate_debug_info();

    options.set_include_callback(|include_request, _, _, _| {
        let mut include_path = shader_path_buf.join(include_request);

        if !include_path.exists() {
            include_path = shader_dir(include_request);
        }

        let include_source = include_structure
            .get_shader_source(&include_path)
            .ok_or_else(|| format!("Including shader: [ {} ] not founded", include_path.display()))?;

        Ok(ResolvedInclude {
            resolved_name: include_request.into(),
            content: include_source,
        })
    });

    let source = include_structure
        .get_shader_source(&shader_path_buf)
        .ok_or_else(|| anyhow!("Compiling shader [ {} ] not founded", shader_path))?;

    let binary_result = SHADER_COMPILER.compile_into_spirv(&source, shader_kind, shader_path, "main", Some(&options))?;

    assert_eq!(Some(&0x07230203), binary_result.as_binary().first());

    Ok(binary_result)
}

pub fn create_pipeline_layout(
    device: &WrappedDevice,
    reflection: &ShaderReflection,
    bindless_descriptor_set_layout: Option<DescriptorSetLayout>,
) -> (PipelineLayout, Vec<DescriptorSetLayout>, Vec<PushConstantRange>) {
    let mut descriptor_set_layouts: Vec<DescriptorSetLayout> = if let Some(bindless_descriptor_set_layout) = bindless_descriptor_set_layout {
        let mut layouts = Vec::with_capacity(reflection.descriptor_template.len() + 1);
        layouts.push(bindless_descriptor_set_layout);
        layouts
    } else {
        Vec::with_capacity(reflection.descriptor_template.len())
    };

    for (set_index, descriptor_set) in &reflection.descriptor_template {
        let descriptor_set_layout_bindings: Vec<DescriptorSetLayoutBinding> = descriptor_set
            .iter()
            .filter_map(|(&binding, descriptor_info)| {
                let binding_count = match descriptor_info.binding_count {
                    BindingCount::One => 1,
                    BindingCount::StaticSized(size) => size as u32,
                    BindingCount::Unbounded => {
                        error!("Descriptor with unbounded count should be used in bindless descriptor set");

                        return None;
                    }
                };

                let descriptor_set_layout_binding = DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(descriptor_set::map_rspirv_descriptor_type(descriptor_info.ty))
                    .descriptor_count(binding_count)
                    .stage_flags(ShaderStageFlags::ALL);

                Some(descriptor_set_layout_binding)
            })
            .collect();

        let descriptor_set_layout_info = DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_set_layout_bindings);

        if let Some(descriptor_set_layout) = unsafe { device.create_descriptor_set_layout(&descriptor_set_layout_info, None).ok() } {
            descriptor_set_layouts.push(descriptor_set_layout);
        } else {
            error!("Failed to create descriptor set layout for set={}", set_index);
        }
    }

    let mut push_constant_ranges: Vec<PushConstantRange> = vec![];

    // Currently only supports a single push constant shared between all shader stages
    if !reflection.push_constant_infos.is_empty() {
        push_constant_ranges.push(
            PushConstantRange::default()
                .size(reflection.push_constant_infos[0].size)
                .offset(reflection.push_constant_infos[0].offset)
                .stage_flags(ShaderStageFlags::ALL),
        );
    }

    let pipeline_layout_info = if !push_constant_ranges.is_empty() {
        PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts).push_constant_ranges(&push_constant_ranges)
    } else {
        PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts)
    };

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None).expect("Failed to create pipeline layout") };

    (pipeline_layout, descriptor_set_layouts, push_constant_ranges)
}

pub fn create_shader_module(device: &WrappedDevice, shader_code: &[u32]) -> Result<ShaderModule> {
    let shader_info = ShaderModuleCreateInfo::default().code(shader_code);

    Ok(unsafe { device.create_shader_module(&shader_info, None) }?)
}
