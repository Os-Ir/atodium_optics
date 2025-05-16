use crate::model::mesh::{MaterialType, MeshBuffer, RenderMaterial, RenderMesh};
use crate::model::vertex::Vertex;
use crate::render_resource::render_buffer::RenderBufferAllocator;
use crate::render_resource::render_image::ImageAllocator;
use crate::render_resource::texture::Texture;
use crate::vk_context::device::WrappedDeviceRef;
use anyhow::{Result, anyhow};
use glam::{Mat4, Vec2, Vec3, Vec4};
use gltf::Node as GltfNode;
use gltf::buffer::Data as GltfBufferData;
use gltf::image::Format as GltfFormat;
use image::{DynamicImage, RgbImage};
use log::{error, info};
use std::mem;

pub mod mesh;
pub mod vertex;

#[derive(Default)]
pub struct RenderModel {
    pub meshes: Vec<(RenderMesh, Mat4)>,
    pub textures: Vec<Texture>,
}

impl RenderModel {
    pub fn new(meshes: Vec<(RenderMesh, Mat4)>, textures: Vec<Texture>) -> Self {
        Self { meshes, textures }
    }

    pub fn merge(&mut self, other: RenderModel) {
        self.meshes.extend(other.meshes);
        self.textures.extend(other.textures);
    }
}

pub fn load_gltf_node(buffer_allocator: &RenderBufferAllocator, node: &GltfNode, buffers: &[GltfBufferData], parent_transform: Mat4) -> Vec<(RenderMesh, Mat4)> {
    let node_transform = parent_transform * Mat4::from_cols_array_2d(&node.transform().matrix());

    let mut meshes = if let Some(mesh) = node.mesh() {
        let primitives = mesh.primitives();

        let mut meshes = Vec::with_capacity(primitives.len());

        for primitive in primitives {
            let reader = primitive.reader(|i| Some(&buffers[i.index()]));

            let indices: Vec<u32> = reader.read_indices().unwrap().into_u32().collect();
            let positions: Vec<Vec3> = reader.read_positions().unwrap().map(Vec3::from).collect();
            let normals: Vec<Vec3> = reader.read_normals().unwrap().map(Vec3::from).collect();

            let tex_coords = if let Some(tex_coords) = reader.read_tex_coords(0) {
                tex_coords.into_f32().map(Vec2::from).collect()
            } else {
                vec![Vec2::new(0.0, 0.0); positions.len()]
            };

            let tangents = if let Some(tangents) = reader.read_tangents() {
                tangents.map(Vec4::from).collect()
            } else {
                vec![Vec4::new(0.0, 0.0, 0.0, 0.0); positions.len()]
            };

            let colors: Vec<_> = if let Some(colors) = reader.read_colors(0) {
                colors.into_rgba_f32().map(Vec4::from).collect()
            } else {
                vec![Vec4::new(1.0, 1.0, 1.0, 1.0); positions.len()]
            };

            let mut vertices: Vec<Vertex> = Vec::with_capacity(positions.len());

            for (i, _) in positions.iter().enumerate() {
                vertices.push(Vertex {
                    pos: positions[i].extend(1.0),
                    normal: normals[i],
                    uv: tex_coords[i],
                    tangent: tangents[i],
                    color: colors[i],
                });
            }

            match MeshBuffer::new(buffer_allocator, indices, vertices) {
                Ok(mesh_buffer) => {
                    let material = primitive.material();
                    let pbr = material.pbr_metallic_roughness();

                    let diffuse_index = pbr.base_color_texture().map_or(u32::MAX, |texture| texture.texture().index() as u32);
                    let normal_index = material.normal_texture().map_or(u32::MAX, |texture| texture.texture().index() as u32);
                    let metallic_roughness_index = pbr.metallic_roughness_texture().map_or(u32::MAX, |texture| texture.texture().index() as u32);
                    let occlusion_index = material.occlusion_texture().map_or(u32::MAX, |texture| texture.texture().index() as u32);

                    let base_color_factor = pbr.base_color_factor();
                    let metallic_factor = pbr.metallic_factor();
                    let roughness_factor = pbr.roughness_factor();

                    let render_material = RenderMaterial {
                        diffuse_map: diffuse_index,
                        normal_map: normal_index,
                        metallic_roughness_map: metallic_roughness_index,
                        occlusion_map: occlusion_index,
                        base_color: Vec4::from(base_color_factor),
                        metallic_factor,
                        roughness_factor,
                        material_type: MaterialType::Lambertian,
                        material_property: 0.0,
                    };

                    meshes.push((RenderMesh::new(mesh_buffer, render_material), node_transform));
                }
                Err(error) => {
                    error!("{}", error);
                }
            }
        }

        meshes
    } else {
        vec![]
    };

    for child in node.children() {
        meshes.extend(load_gltf_node(buffer_allocator, &child, buffers, node_transform));
    }

    meshes
}

pub fn load_gltf(device: WrappedDeviceRef, buffer_allocator: &RenderBufferAllocator, image_allocator: &ImageAllocator, path: &str) -> Result<RenderModel> {
    info!("Loading GLTF model [ {} ]", path);

    let (gltf, buffers, mut images) = gltf::import(path)?;

    let mut textures = Vec::with_capacity(images.len());

    for image in &mut images {
        if image.format == GltfFormat::R8G8B8 {
            let dynamic_image =
                DynamicImage::ImageRgb8(RgbImage::from_raw(image.width, image.height, mem::take(&mut image.pixels)).ok_or_else(|| anyhow!("Failed to create dynamic image from pixel data."))?);

            let rgba8_image = dynamic_image.to_rgba8();
            image.format = GltfFormat::R8G8B8A8;
            image.pixels = rgba8_image.into_raw();
        }

        if image.format != GltfFormat::R8G8B8A8 {
            return Err(anyhow!("Unsupported image format!"));
        }

        let texture = Texture::from_pixels(device.clone(), image_allocator, image.width, image.height, &image.pixels)?;

        textures.push(texture);
    }

    let mut meshes = vec![];
    for scene in gltf.scenes() {
        for node in scene.nodes() {
            meshes.extend(load_gltf_node(buffer_allocator, &node, &buffers, Mat4::IDENTITY));
        }
    }

    Ok(RenderModel::new(meshes, textures))
}
