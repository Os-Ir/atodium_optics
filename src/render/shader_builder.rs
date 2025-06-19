use log::info;
use spirv_builder::{Capability, MetadataPrintout, SpirvBuilder};
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

#[inline]
pub fn shader_base_dir() -> PathBuf {
    crate::util::lib_root().join("shaders")
}

pub type SpirvShaders = HashMap<String, SpirvShader>;

#[derive(Debug, Clone)]
pub struct SpirvShader {
    pub name: String,
    pub binary: Vec<u32>,
    pub binary_u8: Vec<u8>,
}

impl SpirvShader {
    pub fn new(name: String, binary: Vec<u32>) -> Self {
        let binary_u8 = binary.iter().flat_map(|word| word.to_le_bytes().to_vec()).collect();

        SpirvShader {
            name: name.clone(),
            binary,
            binary_u8,
        }
    }

    #[inline]
    pub fn as_binary(&self) -> &[u32] {
        &self.binary
    }

    #[inline]
    pub fn as_binary_u8(&self) -> &[u8] {
        &self.binary_u8
    }
}

pub fn compile_spirv_shaders() -> HashMap<String, SpirvShader> {
    info!("Compiling spirv shaders");

    SpirvBuilder::new(shader_base_dir(), "spirv-unknown-vulkan1.1")
        .print_metadata(MetadataPrintout::None)
        .shader_panic_strategy(spirv_builder::ShaderPanicStrategy::DebugPrintfThenExit {
            print_inputs: true,
            print_backtrace: true,
        })
        .multimodule(true)
        .capability(Capability::RayQueryKHR)
        .capability(Capability::RayTracingKHR)
        .extension("SPV_KHR_ray_query")
        .extension("SPV_KHR_ray_tracing")
        .build()
        .unwrap()
        .module
        .unwrap_multi()
        .iter()
        .map(|(name, path)| {
            let mut shader_file = File::open(path).unwrap();
            let shader = SpirvShader::new(name.clone(), ash::util::read_spv(&mut shader_file).unwrap());

            (name.clone(), shader)
        })
        .collect()
}
