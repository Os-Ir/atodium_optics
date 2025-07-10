use crate::camera::film::Film;
use crate::light::medium::Medium;
use crate::light::ray::{Ray, RayDifferential};
use crate::spectrum::{SampledSpectrum, SampledWavelengths};
use crate::util::frame::Frame;
use crate::util::{math, sampling};
use core::ops::Deref;
use spirv_std::glam::{Mat4, Quat, Vec2, Vec3};

pub mod film;
pub mod filter;
pub mod transform;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct CameraSample {
    pub point_film: Vec2,
    pub point_lens: Vec2,
    pub time: f32,
    pub filer_weight: f32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct CameraRay {
    pub ray: Ray,
    pub weight: SampledSpectrum,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct CameraRayDifferential {
    pub ray: RayDifferential,
    pub weight: SampledSpectrum,
}

pub trait ICamera {
    fn gen_ray(&self, sample: CameraSample, lambda: SampledWavelengths) -> Option<CameraRay>;

    fn gen_ray_differential(&self, sample: CameraSample, lambda: SampledWavelengths) -> Option<CameraRayDifferential> {
        let camera_ray = self.gen_ray(sample, lambda)?;

        let rx = [-0.05, 0.05].into_iter().find_map(|eps| {
            let mut shift_sample = sample;
            shift_sample.point_film.x += eps;

            if let Some(ray_shift) = self.gen_ray(shift_sample, lambda) {
                let rx_origin = camera_ray.ray.origin + (ray_shift.ray.origin - camera_ray.ray.origin) / eps;
                let rx_direction = camera_ray.ray.direction + (ray_shift.ray.direction - camera_ray.ray.direction) / eps;

                Some((rx_origin, rx_direction))
            } else {
                None
            }
        });

        let ry = [-0.05, 0.05].into_iter().find_map(|eps| {
            let mut shift_sample = sample;
            shift_sample.point_film.y += eps;

            if let Some(ray_shift) = self.gen_ray(shift_sample, lambda) {
                let ry_origin = camera_ray.ray.origin + (ray_shift.ray.origin - camera_ray.ray.origin) / eps;
                let ry_direction = camera_ray.ray.direction + (ray_shift.ray.direction - camera_ray.ray.direction) / eps;

                Some((ry_origin, ry_direction))
            } else {
                None
            }
        });

        let (rx_origin, rx_direction) = match rx {
            Some((origin, direction)) => (origin, direction),
            None => (camera_ray.ray.origin, camera_ray.ray.direction),
        };

        let (ry_origin, ry_direction) = match ry {
            Some((origin, direction)) => (origin, direction),
            None => (camera_ray.ray.origin, camera_ray.ray.direction),
        };

        let ray = RayDifferential {
            base: camera_ray.ray,
            has_differentials: rx.is_some() && ry.is_some(),
            rx_origin,
            ry_origin,
            rx_direction,
            ry_direction,
        };

        Some(CameraRayDifferential { ray, weight: camera_ray.weight })
    }

    fn get_film(&self) -> &Film;

    fn get_camera_transform(&self) -> Mat4;

    fn sample_time(&self, u: f32) -> f32;
}

#[derive(Clone)]
pub enum Camera {
    Perspective(PerspectiveCamera),
}

impl Deref for Camera {
    type Target = dyn ICamera;

    fn deref(&self) -> &Self::Target {
        match self {
            Camera::Perspective(camera) => camera,
        }
    }
}

#[derive(Clone)]
#[repr(C)]
pub struct CameraBaseParameters {
    pub camera_transform: Mat4,
    pub shutter_open: f32,
    pub shutter_close: f32,
    pub film: Film,
    pub medium: Option<Medium>,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct MinDifferentials {
    pub min_pos_differential_x: Vec3,
    pub min_pos_differential_y: Vec3,
    pub min_dir_differential_x: Vec3,
    pub min_dir_differential_y: Vec3,
}

impl MinDifferentials {
    pub fn new(camera: &dyn ICamera, film: &Film) -> Self {
        let mut min_pos_differential_x = Vec3::INFINITY;
        let mut min_pos_differential_y = Vec3::INFINITY;
        let mut min_dir_differential_x = Vec3::INFINITY;
        let mut min_dir_differential_y = Vec3::INFINITY;

        let mut sample = CameraSample {
            point_film: Vec2::ZERO,
            point_lens: Vec2::new(0.5, 0.5),
            time: 0.5,
            filer_weight: 1.0,
        };

        let lambda = SampledWavelengths::sample_visible(0.5);

        let n: i32 = 512;
        for i in 0..n {
            sample.point_film.x = (i as f32) / (n as f32 - 1.0) * (film.full_resolution().x as f32);
            sample.point_film.y = (i as f32) / (n as f32 - 1.0) * (film.full_resolution().y as f32);

            if let Some(ray_differential) = camera.gen_ray_differential(sample, lambda) {
                let mut ray = ray_differential.ray;

                let dox = camera.get_camera_transform().inverse().transform_point3(ray.rx_origin - ray.origin);
                let doy = camera.get_camera_transform().inverse().transform_point3(ray.ry_origin - ray.origin);

                if dox.length_squared() < min_pos_differential_x.length_squared() {
                    min_pos_differential_x = dox;
                }
                if dox.length_squared() < min_pos_differential_y.length_squared() {
                    min_pos_differential_y = doy;
                }

                ray.base.direction = ray.direction.normalize();
                ray.rx_direction = ray.rx_direction.normalize();
                ray.ry_direction = ray.ry_direction.normalize();

                let frame = Frame::from_z(ray.direction);

                let df = frame.global_to_local(ray.direction);
                let dxf = frame.global_to_local(ray.rx_direction).normalize();
                let dyf = frame.global_to_local(ray.ry_direction).normalize();

                if (dxf - df).length_squared() < min_dir_differential_x.length_squared() {
                    min_dir_differential_x = dxf - df;
                }
                if (dyf - df).length_squared() < min_dir_differential_y.length_squared() {
                    min_dir_differential_y = dyf - df;
                }
            }
        }

        Self {
            min_pos_differential_x,
            min_pos_differential_y,
            min_dir_differential_x,
            min_dir_differential_y,
        }
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct ProjectiveCamera {
    pub screen_from_camera: Mat4,
    pub camera_from_raster: Mat4,
    pub raster_from_screen: Mat4,
    pub screen_from_raster: Mat4,
    pub lens_radius: f32,
    pub focal_distance: f32,
}

impl ProjectiveCamera {
    pub fn new(params: &CameraBaseParameters, screen_from_camera: Mat4, window_min: Vec2, window_max: Vec2, lens_radius: f32, focal_distance: f32) -> Self {
        let ndc_from_screen = Mat4::from_scale_rotation_translation(
            Vec3::new(1.0 / (window_max.x - window_min.x), 1.0 / (window_max.y - window_min.y), 1.0),
            Quat::IDENTITY,
            Vec3::new(-window_min.x, -window_max.y, 0.0),
        );

        let full_resolution = params.film.full_resolution();

        let raster_from_ndc = Mat4::from_scale(Vec3::new(full_resolution.x as f32, -(full_resolution.y as f32), 1.0));

        let raster_from_screen = raster_from_ndc * ndc_from_screen;
        let screen_from_raster = raster_from_screen.inverse();

        let camera_from_raster = screen_from_camera.inverse() * screen_from_raster;

        Self {
            screen_from_camera,
            camera_from_raster,
            raster_from_screen,
            screen_from_raster,
            lens_radius,
            focal_distance,
        }
    }
}

#[derive(Clone)]
#[repr(C)]
pub struct PerspectiveCamera {
    pub params: CameraBaseParameters,
    pub base: ProjectiveCamera,
    pub dx_camera: Vec3,
    pub dy_camera: Vec3,
    pub cos_total_width: f32,
    pub area: f32,
    pub min_differentials: Option<MinDifferentials>,
}

impl PerspectiveCamera {
    pub fn new(params: CameraBaseParameters, fov: f32, window_min: Vec2, window_max: Vec2, lens_radius: f32, focal_distance: f32) -> Self {
        let base = ProjectiveCamera::new(&params, math::perspective(fov, 0.01, 1000.0), window_min, window_max, lens_radius, focal_distance);

        let dx_camera = base.camera_from_raster.transform_point3(Vec3::new(1.0, 0.0, 0.0)) - base.camera_from_raster.transform_point3(Vec3::ZERO);
        let dy_camera = base.camera_from_raster.transform_point3(Vec3::new(0.0, 1.0, 0.0)) - base.camera_from_raster.transform_point3(Vec3::ZERO);

        let radius = params.film.get_filter().radius();
        let p_corner = Vec3::new(-radius.x, -radius.y, 0.0);

        let w_corner_camera = base.camera_from_raster.transform_point3(p_corner).normalize();
        let cos_total_width = w_corner_camera.z;

        let resolution = params.film.full_resolution();

        let mut point_min = base.camera_from_raster.transform_point3(Vec3::ZERO);
        let mut point_max = base.camera_from_raster.transform_point3(Vec3::new(resolution.x as _, resolution.y as _, 0.0));
        point_min /= point_min.z;
        point_max /= point_max.z;
        let area = (point_max.x - point_min.x) * (point_max.y - point_min.y);

        let mut camera = Self {
            params,
            base,
            dx_camera,
            dy_camera,
            cos_total_width,
            area,
            min_differentials: None,
        };

        let min_differentials = MinDifferentials::new(&camera, &camera.params.film);
        camera.min_differentials = Some(min_differentials);

        camera
    }
}

impl Deref for PerspectiveCamera {
    type Target = ProjectiveCamera;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl ICamera for PerspectiveCamera {
    fn gen_ray(&self, sample: CameraSample, _: SampledWavelengths) -> Option<CameraRay> {
        let point_film = Vec3::new(sample.point_film.x, sample.point_film.y, 0.0);
        let point_camera = self.camera_from_raster.transform_point3(point_film).normalize();

        let mut ray = Ray::new(Vec3::ZERO, point_camera, sample.time, self.params.medium);

        if self.lens_radius > 0.0 {
            let point_lens = self.lens_radius * sampling::sample_uniform_disk_concentric(sample.point_lens);

            let focal_t = self.focal_distance / ray.direction.z;
            let point_focus = ray.at(focal_t);

            ray.origin = Vec3::new(point_lens.x, point_lens.y, 0.0);
            ray.direction = (point_focus - ray.origin).normalize();
        }

        Some(CameraRay {
            ray: ray.transform(self.get_camera_transform()),
            weight: SampledSpectrum::uniform(1.0),
        })
    }

    fn gen_ray_differential(&self, sample: CameraSample, _: SampledWavelengths) -> Option<CameraRayDifferential> {
        let point_film = Vec3::new(sample.point_film.x, sample.point_film.y, 0.0);
        let point_camera = self.camera_from_raster.transform_point3(point_film).normalize();

        let mut ray: RayDifferential = Ray::new(Vec3::ZERO, point_camera, sample.time, self.params.medium).into();

        if self.lens_radius > 0.0 {
            let point_lens = self.lens_radius * sampling::sample_uniform_disk_concentric(sample.point_lens);

            let focal_t = self.focal_distance / ray.direction.z;
            let point_focus = ray.at(focal_t);
            ray.base.origin = Vec3::new(point_lens.x, point_lens.y, 0.0);
            ray.base.direction = (point_focus - ray.origin).normalize();

            let dx = (point_camera + self.dx_camera).normalize();
            let focal_t = self.focal_distance / dx.z;
            let point_focus = dx * focal_t;
            ray.rx_origin = Vec3::new(point_lens.x, point_lens.y, 0.0);
            ray.rx_direction = (point_focus - ray.rx_origin).normalize();

            let dy = (point_camera + self.dy_camera).normalize();
            let focal_t = self.focal_distance / dy.z;
            let point_focus = dy * focal_t;
            ray.ry_origin = Vec3::new(point_lens.x, point_lens.y, 0.0);
            ray.ry_direction = (point_focus - ray.ry_origin).normalize();
        } else {
            ray.rx_origin = ray.origin;
            ray.ry_origin = ray.origin;
            ray.rx_direction = (point_camera + self.dx_camera).normalize();
            ray.ry_direction = (point_camera + self.dy_camera).normalize();
        }

        ray.has_differentials = true;

        Some(CameraRayDifferential {
            ray,
            weight: SampledSpectrum::uniform(1.0),
        })
    }

    fn get_film(&self) -> &Film {
        &self.params.film
    }

    fn get_camera_transform(&self) -> Mat4 {
        self.params.camera_transform
    }

    fn sample_time(&self, u: f32) -> f32 {
        math::lerp(u, self.params.shutter_open, self.params.shutter_close)
    }
}
