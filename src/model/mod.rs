use crate::memory::render_buffer::{RenderBuffer, RenderBufferAllocator};
use crate::memory::render_image::ImageAllocator;
use crate::memory::texture::Texture;
use crate::model::mesh::{MaterialType, MeshBuffer, RenderMaterial, RenderMesh};
use crate::model::vertex::Vertex;
use crate::render::device::WrappedDeviceRef;
use crate::rt::blas;
use crate::rt::blas::Blas;
use crate::rt::tlas::InstanceMetadata;
use anyhow::{anyhow, bail, Result};
use ash::vk::BufferUsageFlags;
use glam::{Mat4, Vec2, Vec3, Vec4};
use gltf::buffer::Data as GltfBufferData;
use gltf::image::Format as GltfFormat;
use gltf::Node as GltfNode;
use gpu_allocator::MemoryLocation;
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

    pub fn write_vertices_to_buffer(&self, allocator: &RenderBufferAllocator) -> Result<RenderBuffer> {
        let vertices = self.meshes.iter().map(|(mesh, _)| mesh.mesh_buffer.vertices.clone()).flatten().collect::<Vec<_>>();

        let vertices_buffer = allocator.allocate(
            (vertices.len() * mem::size_of::<Vertex>()) as _,
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;

        allocator.upload_data(&vertices_buffer, &vertices)?;

        Ok(vertices_buffer)
    }

    pub fn write_indices_to_buffer(&self, allocator: &RenderBufferAllocator) -> Result<RenderBuffer> {
        let mut current_index = 0_u32;
        let mut indices = Vec::new();

        for (mesh, _) in &self.meshes {
            indices.append(&mut mesh.mesh_buffer.indices.iter().map(|&idx| idx + current_index).collect::<Vec<u32>>());
            current_index += mesh.mesh_buffer.vertices.len() as u32;
        }

        let indices_buffer = allocator.allocate(
            (indices.len() * mem::size_of::<u32>()) as _,
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;

        allocator.upload_data(&indices_buffer, &indices)?;

        Ok(indices_buffer)
    }

    pub fn write_instance_metadata_to_buffer(&self, allocator: &RenderBufferAllocator) -> Result<RenderBuffer> {
        let mut current_index = 0_u32;
        let mut metadata = Vec::new();

        for (mesh, transform) in &self.meshes {
            metadata.push(InstanceMetadata {
                transform: *transform,
                index_offset: current_index,
            });

            current_index += mesh.mesh_buffer.indices.len() as u32;
        }

        let metadata_buffer = allocator.allocate(
            (metadata.len() * mem::size_of::<InstanceMetadata>()) as _,
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;

        allocator.upload_data(&metadata_buffer, &metadata)?;

        Ok(metadata_buffer)
    }

    pub fn write_material_to_buffer(&self, allocator: &RenderBufferAllocator) -> Result<RenderBuffer> {
        let materials = self.meshes.iter().map(|(mesh, _)| mesh.material).collect::<Vec<_>>();

        let material_buffer = allocator.allocate(
            (materials.len() * mem::size_of::<RenderMaterial>()) as _,
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;

        allocator.upload_data(&material_buffer, &materials)?;

        Ok(material_buffer)
    }

    pub fn build_blas(&self, device: WrappedDeviceRef, allocator: &RenderBufferAllocator) -> Vec<Blas> {
        self.meshes
            .iter()
            .enumerate()
            .filter_map(|(i, (mesh, _))| match blas::create_blas(device.clone(), &allocator, &mesh.mesh_buffer, i as _) {
                Ok(blas) => Some(blas),
                Err(error) => {
                    error!("Failed to build bottom level acceleration structure: {:?}", error);
                    None
                }
            })
            .collect::<Vec<_>>()
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
                        base_color: Vec4::from(base_color_factor),
                        diffuse_map: diffuse_index,
                        normal_map: normal_index,
                        metallic_roughness_map: metallic_roughness_index,
                        occlusion_map: occlusion_index,
                        metallic_factor,
                        roughness_factor,
                        material_type: MaterialType::default().into(),
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
            bail!("Unsupported image format!");
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
