use crate::light::medium::{Medium, MediumInterface, PhaseFunction};
use crate::light::ray;
use crate::light::ray::{Ray, RayDifferential, Vec3i};
use core::ops::Deref;
use spirv_std::glam::{Vec2, Vec3};

#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct Interaction {
    pub point: Vec3i,
    pub output_direction: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
    pub time: f32,
    pub medium_interface: MediumInterface,
    pub medium: Option<Medium>,
}

impl Interaction {
    #[inline]
    pub fn is_surface_interaction(&self) -> bool {
        self.normal.x != 0.0 || self.normal.y != 0.0 || self.normal.z != 0.0
    }

    #[inline]
    pub fn is_medium_interaction(&self) -> bool {
        !self.is_surface_interaction()
    }

    pub fn spawn_ray(&self, direction: Vec3) -> RayDifferential {
        let origin = ray::offset_ray_origin(self.point, self.normal, direction);
        Ray::new(origin, direction, self.time, self.medium).into()
    }

    pub fn spawn_ray_to(&self, target: Vec3) -> RayDifferential {
        let mut ray = ray::spawn_ray_to(self.point, self.normal, self.time, target);
        ray.medium = self.medium;
        ray.into()
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct MediumInteraction {
    pub base: Interaction,
    pub phase: PhaseFunction,
}

impl Deref for MediumInteraction {
    type Target = Interaction;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl Into<Interaction> for MediumInteraction {
    fn into(self) -> Interaction {
        self.base
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct SurfaceInteraction {
    pub base: Interaction,
    pub geometry_partial_point_u: Vec3,
    pub geometry_partial_point_v: Vec3,
    pub geometry_partial_normal_u: Vec3,
    pub geometry_partial_normal_v: Vec3,
    pub shading_partial_point_u: Vec3,
    pub shading_partial_point_v: Vec3,
    pub shading_partial_normal_u: Vec3,
    pub shading_partial_normal_v: Vec3,
    pub partial_point_x: Vec3,
    pub partial_point_y: Vec3,
    pub partial_u_x: Vec3,
    pub partial_u_y: Vec3,
    pub partial_v_x: Vec3,
    pub partial_v_y: Vec3,
    pub face_index: u32,
    // TODO: material, area_light,
}

impl Deref for SurfaceInteraction {
    type Target = Interaction;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl Into<Interaction> for SurfaceInteraction {
    fn into(self) -> Interaction {
        self.base
    }
}
