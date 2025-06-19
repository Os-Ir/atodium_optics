use spirv_std::glam::Vec3;

#[derive(Copy, Clone)]
pub struct Frame {
    pub x: Vec3,
    pub y: Vec3,
    pub z: Vec3,
}

impl Frame {
    pub fn new() -> Self {
        Self { x: Vec3::X, y: Vec3::Y, z: Vec3::Z }
    }

    pub fn new_with_vectors(x: Vec3, y: Vec3, z: Vec3) -> Self {
        Self { x, y, z }
    }

    pub fn from_xz(x: Vec3, z: Vec3) -> Self {
        Self::new_with_vectors(x, z.cross(x), z)
    }

    pub fn from_xy(x: Vec3, y: Vec3) -> Self {
        Self::new_with_vectors(x, y, x.cross(y))
    }

    pub fn from_z(z: Vec3) -> Self {
        let (x, y) = Self::coordinate_system(z);
        Self::new_with_vectors(x, y, z)
    }

    pub fn from_x(x: Vec3) -> Self {
        let (y, z) = Self::coordinate_system(x);
        Self::new_with_vectors(x, y, z)
    }

    pub fn from_y(y: Vec3) -> Self {
        let (z, x) = Self::coordinate_system(y);
        Self::new_with_vectors(x, y, z)
    }

    pub fn global_to_local(&self, v: Vec3) -> Vec3 {
        Vec3::new(v.dot(self.x), v.dot(self.y), v.dot(self.z))
    }

    pub fn local_to_global(&self, v: Vec3) -> Vec3 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    fn coordinate_system(v1: Vec3) -> (Vec3, Vec3) {
        let sign = if v1.z >= 0.0 { 1.0 } else { -1.0 };
        let a = -1.0 / (sign + v1.z);
        let b = v1.x * v1.y * a;

        let v2 = Vec3::new(1.0 + sign * v1.x * v1.x * a, sign * b, -sign * v1.x);

        let v3 = Vec3::new(b, sign + v1.y * v1.y * a, -v1.y);

        (v2, v3)
    }
}

impl Default for Frame {
    fn default() -> Self {
        Self::new()
    }
}
