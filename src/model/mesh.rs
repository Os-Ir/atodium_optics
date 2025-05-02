use crate::model::vertex::Vertex;
use crate::render_resource::render_buffer::{RenderBuffer, RenderBufferAllocator};
use anyhow::Result;
use ash::vk::{BufferUsageFlags, DeviceSize};
use gpu_allocator::MemoryLocation;

pub struct MeshBuffer {
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
    pub index_buffer: RenderBuffer,
    pub vertex_buffer: RenderBuffer,
}

impl MeshBuffer {
    pub fn new(allocator: &RenderBufferAllocator, indices: Vec<u32>, vertices: Vec<Vertex>) -> Result<Self> {
        let index_buffer = allocator.allocate(
            size_of_val(&indices) as DeviceSize,
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuOnly,
        )?;

        allocator.upload_data(&index_buffer, &indices)?;

        let vertex_buffer = allocator.allocate(
            size_of_val(&vertices) as DeviceSize,
            BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::SHADER_DEVICE_ADDRESS | BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            MemoryLocation::GpuOnly,
        )?;

        allocator.upload_data(&vertex_buffer, &vertices)?;

        Ok(Self {
            indices,
            vertices,
            index_buffer,
            vertex_buffer,
        })
    }
}

#[derive(Copy, Clone)]
pub enum MaterialType {
    Lambertian,
    Metal,
    Dielectric,
    DiffuseLight,
}

impl Default for MaterialType {
    fn default() -> Self {
        Self::Lambertian
    }
}

#[derive(Default, Copy, Clone)]
pub struct Material {
    pub diffuse_map: u32,
    pub normal_map: u32,
    pub metallic_roughness_map: u32,
    pub occlusion_map: u32,
    pub base_color: [f32; 3],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub material_type: MaterialType,
    pub material_property: f32,
}

pub struct RenderMesh {
    pub mesh_buffer: MeshBuffer,
    pub material: Material,
}
