use crate::light::medium::Medium;
use crate::util;
use core::ops::Deref;
use spirv_std::glam::{Mat4, Vec3};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    pub time: f32,
    pub medium: Option<Medium>,
}

impl Ray {
    pub unsafe fn new_unchecked(origin: Vec3, direction: Vec3, time: f32, medium: Option<Medium>) -> Self {
        Self { origin, direction, time, medium }
    }

    pub fn new(origin: Vec3, direction: Vec3, time: f32, medium: Option<Medium>) -> Self {
        unsafe { Self::new_unchecked(origin, direction.normalize(), time, medium) }
    }

    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    pub fn transform(&self, transform: Mat4) -> Self {
        let new_origin = transform.transform_point3(self.origin);
        let new_direction = transform.transform_vector3(self.direction).normalize();

        Self::new(new_origin, new_direction, self.time, self.medium)
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RayDifferential {
    pub base: Ray,
    pub has_differentials: bool,
    pub rx_origin: Vec3,
    pub ry_origin: Vec3,
    pub rx_direction: Vec3,
    pub ry_direction: Vec3,
}

impl Deref for RayDifferential {
    type Target = Ray;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl From<Ray> for RayDifferential {
    fn from(ray: Ray) -> Self {
        Self::new(ray)
    }
}

impl RayDifferential {
    pub fn new(ray: Ray) -> Self {
        Self {
            base: ray,
            has_differentials: false,
            rx_origin: Vec3::ZERO,
            ry_origin: Vec3::ZERO,
            rx_direction: Vec3::ZERO,
            ry_direction: Vec3::ZERO,
        }
    }

    pub fn scale_differentials(&mut self, scale: f32) {
        self.rx_origin = self.origin + (self.rx_origin - self.origin) * scale;
        self.ry_origin = self.origin + (self.ry_origin - self.origin) * scale;
        self.rx_direction = self.direction + (self.rx_direction - self.direction) * scale;
        self.ry_direction = self.direction + (self.ry_direction - self.direction) * scale;
    }

    pub fn transform(&self, transform: Mat4) -> Self {
        let new_base = self.base.transform(transform);
        let new_rx_origin = transform.transform_point3(self.rx_origin);
        let new_ry_origin = transform.transform_point3(self.ry_origin);
        let new_rx_direction = transform.transform_vector3(self.rx_direction).normalize();
        let new_ry_direction = transform.transform_vector3(self.ry_direction).normalize();

        Self {
            base: new_base,
            has_differentials: self.has_differentials,
            rx_origin: new_rx_origin,
            ry_origin: new_ry_origin,
            rx_direction: new_rx_direction,
            ry_direction: new_ry_direction,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct Vec3i {
    pub point: Vec3,
    pub error: Vec3,
}

impl From<Vec3> for Vec3i {
    fn from(point: Vec3) -> Self {
        Self { point, error: Vec3::ZERO }
    }
}

#[inline]
pub fn offset_ray_origin(point: Vec3i, normal: Vec3, direction: Vec3) -> Vec3 {
    let d = normal.abs().dot(point.error);
    let mut offset = normal * d;
    if direction.dot(normal) < 0.0 {
        offset = -offset;
    }

    let mut moved = point.point + offset;

    for i in 0..3 {
        if offset[i] > 0.0 {
            moved[i] = util::next_float_up(moved[i]);
        } else if offset[i] < 0.0 {
            moved[i] = util::next_float_down(moved[i]);
        }
    }

    moved
}

#[inline]
pub fn spawn_ray(point: Vec3i, normal: Vec3, time: f32, direction: Vec3) -> Ray {
    Ray::new(offset_ray_origin(point, normal, direction), direction, time, None)
}

#[inline]
pub fn spawn_ray_to(point_from: Vec3i, normal: Vec3, time: f32, point_to: Vec3) -> Ray {
    spawn_ray(point_from, normal, time, point_to - point_from.point)
}
