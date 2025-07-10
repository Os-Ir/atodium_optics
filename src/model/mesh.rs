use crate::memory::render_buffer::{RenderBuffer, RenderBufferAllocator};
use crate::model::vertex::Vertex;
use anyhow::Result;
use ash::vk::{BufferUsageFlags, DeviceSize};
use glam::Vec4;
use gpu_allocator::MemoryLocation;
use std::mem;

pub struct MeshBuffer {
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
    pub index_buffer: RenderBuffer,
    pub vertex_buffer: RenderBuffer,
}

impl MeshBuffer {
    pub fn new(allocator: &RenderBufferAllocator, indices: Vec<u32>, vertices: Vec<Vertex>) -> Result<Self> {
        let index_buffer = allocator.allocate(
            (indices.len() * mem::size_of::<u32>()) as DeviceSize,
            BufferUsageFlags::STORAGE_BUFFER
                | BufferUsageFlags::TRANSFER_DST
                | BufferUsageFlags::INDEX_BUFFER
                | BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            MemoryLocation::GpuOnly,
        )?;

        allocator.upload_data::<u32>(&index_buffer, &indices)?;

        let vertex_buffer = allocator.allocate(
            (vertices.len() * mem::size_of::<Vertex>()) as DeviceSize,
            BufferUsageFlags::STORAGE_BUFFER
                | BufferUsageFlags::TRANSFER_DST
                | BufferUsageFlags::VERTEX_BUFFER
                | BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            MemoryLocation::GpuOnly,
        )?;

        allocator.upload_data::<Vertex>(&vertex_buffer, &vertices)?;

        Ok(Self {
            indices,
            vertices,
            index_buffer,
            vertex_buffer,
        })
    }
}

#[derive(Copy, Clone, Debug)]
pub enum MaterialType {
    Lambertian,
    Conductor,
    Dielectric,
}

impl Default for MaterialType {
    fn default() -> Self {
        Self::Lambertian
    }
}

impl Into<u32> for MaterialType {
    fn into(self) -> u32 {
        match self {
            MaterialType::Lambertian => 0,
            MaterialType::Conductor => 1,
            MaterialType::Dielectric => 2,
        }
    }
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
pub struct RenderMaterial {
    pub base_color: Vec4,
    pub metallic_factor: f32,
    pub roughness_factor: f32,

    pub diffuse_map: u32,
    pub normal_map: u32,
    pub metallic_roughness_map: u32,
    pub occlusion_map: u32,

    pub material_type: u32,
    pub material_property: f32,
}

pub struct RenderMesh {
    pub mesh_buffer: MeshBuffer,
    pub material: RenderMaterial,
}

impl RenderMesh {
    pub fn new(mesh_buffer: MeshBuffer, material: RenderMaterial) -> Self {
        Self { mesh_buffer, material }
    }
}
