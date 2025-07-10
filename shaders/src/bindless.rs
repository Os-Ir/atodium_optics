use core::ops::Index;
use spirv_std::glam::{Mat4, Vec2, Vec3, Vec4};

#[derive(Default, Clone, Copy)]
#[repr(C)]
pub struct InstanceMetadata {
    pub transform: Mat4,
    pub index_offset: u32,
}

#[derive(Default, Clone, Copy)]
#[repr(C)]
pub struct Vertex {
    pub pos: Vec4,
    pub normal: Vec3,
    pub uv: Vec2,
    pub color: Vec4,
    pub tangent: Vec4,
}

#[derive(Copy, Clone, Debug)]
pub enum MaterialType {
    Lambertian,
    Metal,
    Dielectric,
}

impl From<u32> for MaterialType {
    fn from(value: u32) -> Self {
        match value {
            0 => MaterialType::Lambertian,
            1 => MaterialType::Metal,
            2 => MaterialType::Dielectric,
            _ => panic!("Invalid material type"),
        }
    }
}

#[derive(Clone, Copy)]
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

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Light {
    pub pos: Vec3,
    pub scale: f32,
    // TODO: extend this from point light into full light struct
}

#[inline]
pub fn get_instance_metadata(metadata: &[InstanceMetadata], instance_custom_index: u32) -> InstanceMetadata {
    metadata[instance_custom_index as usize]
}

#[inline]
pub fn get_global_index_offset(metadata: &[InstanceMetadata], instance_custom_index: u32, primitive_id: u32) -> usize {
    let instance_index_offset = get_instance_metadata(metadata, instance_custom_index).index_offset;
    (instance_index_offset + 3 * primitive_id) as _
}

#[inline]
pub fn get_instance_material(materials: &[RenderMaterial], instance_custom_index: u32) -> &RenderMaterial {
    materials.index(instance_custom_index as usize)
}
