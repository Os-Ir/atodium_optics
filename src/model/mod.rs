use crate::model::mesh::RenderMesh;
use crate::render_resource::texture::Texture;
use glam::Mat4;

pub mod mesh;
pub mod vertex;

#[derive(Default)]
pub struct RenderModel {
    pub meshes: Vec<(RenderMesh, Mat4)>,
    pub textures: Vec<Texture>,
    pub model_transform: Mat4,
}

impl RenderModel {
    pub fn new(meshes: Vec<(RenderMesh, Mat4)>, textures: Vec<Texture>) -> Self {
        Self {
            meshes,
            textures,
            model_transform: Mat4::IDENTITY,
        }
    }
}
