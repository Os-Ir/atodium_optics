use glam::{Mat4, Vec2, Vec3, Vec4};

#[derive(Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct Vertex {
    pub pos: Vec4,
    pub normal: Vec3,
    pub uv: Vec2,
    pub color: Vec4,
    pub tangent: Vec4,
}

impl Vertex {
    pub fn new(pos: Vec4, normal: Vec3, uv: Vec2, color: Vec4, tangent: Vec4) -> Self {
        Self { pos, normal, uv, color, tangent }
    }

    pub fn transform(&self, trans: Mat4) -> Self {
        Self::new(trans.mul_vec4(self.pos), trans.transform_point3(self.normal), self.uv, self.color, self.tangent)
    }
}
