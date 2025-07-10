use crate::light::interaction::Interaction;
use crate::light::medium::MediumInterface;
use crate::light::ray::Ray;
use crate::spectrum::{DenselySampledSpectrum, ISpectrum, SampledSpectrum, SampledWavelengths};
use crate::util::sampling;
use core::f32::consts;
use core::ops::Deref;
use spirv_std::glam::{Mat4, Vec2, Vec3};

pub mod interaction;
pub mod medium;
pub mod ray;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct LightRadianceInputSample {
    pub radiance: SampledSpectrum,
    pub interaction: Interaction,
    pub input_direction: Vec3,
    pub pdf: f32,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct LightRadianceEmittedSample {
    pub radiance: SampledSpectrum,
    pub interaction: Option<Interaction>,
    pub ray: Ray,
    pub pdf_position: f32,
    pub pdf_direction: f32,
}

#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct LightSampleContext {
    pub point: Vec3,
    pub geometry_normal: Vec3,
    pub shading_normal: Vec3,
}

#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct LightBounds {
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub direction: Vec3,
    pub emitted_power: f32,
    pub cos_theta_o: f32,
    pub cos_theta_e: f32,
    pub two_sided: bool,
}

pub trait ILight {
    fn total_emitted_power(&self, lambda: &SampledWavelengths) -> SampledSpectrum;

    fn sample_radiance_input(&self, ctx: LightSampleContext, u: Vec2, lambda: &SampledWavelengths, allow_incomplete_pdf: bool) -> Option<LightRadianceInputSample>;

    fn pdf_radiance_input(&self, ctx: LightSampleContext, input_direction: Vec3, allow_incomplete_pdf: bool) -> f32;

    fn sample_radiance_emitted(&self, u1: Vec2, u2: Vec2, lambda: &SampledWavelengths, time: f32) -> Option<LightRadianceEmittedSample>;

    fn pdf_radiance_emitted(&self, ray: Ray) -> (f32, f32);

    fn radiance(&self, point: Vec3, normal: Vec3, uv: Vec2, direction: Vec3, lambda: &SampledWavelengths) -> SampledSpectrum;

    fn radiance_emitted(&self, ray: Ray, lambda: &SampledWavelengths) -> SampledSpectrum;

    fn preprocess(&mut self, scene_bounds_min: Vec3, scene_bounds_max: Vec3);
}

#[derive(Copy, Clone)]
pub enum Light {
    Point(PointLight),
    Distant,
    Projection,
    Goniometric,
    Spot,
    DiffuseArea,
    UniformInfinite,
    ImageInfinite,
    PortalImageInfinite,
}

impl Deref for Light {
    type Target = dyn ILight;

    fn deref(&self) -> &Self::Target {
        match self {
            Light::Point(light) => light,
            _ => todo!(),
        }
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct LightBase {
    pub render_from_light: Mat4,
    pub medium_interface: MediumInterface,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct PointLight {
    base: LightBase,
    intensity: DenselySampledSpectrum,
    scale: f32,
}

impl Deref for PointLight {
    type Target = LightBase;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl ILight for PointLight {
    fn total_emitted_power(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        self.intensity.sample(&lambda) * self.scale * 4.0 * consts::PI
    }

    fn sample_radiance_input(&self, ctx: LightSampleContext, _: Vec2, lambda: &SampledWavelengths, _: bool) -> Option<LightRadianceInputSample> {
        let point = self.render_from_light.transform_point3(Vec3::ZERO);
        let input_direction = (point - ctx.point).normalize();
        let radiance = self.intensity.sample(lambda) * self.scale / (point - ctx.point).length_squared();

        Some(LightRadianceInputSample {
            radiance,
            interaction: Interaction {
                point: point.into(),
                medium_interface: self.medium_interface,
                ..Default::default()
            },
            input_direction,
            pdf: 1.0,
        })
    }

    fn pdf_radiance_input(&self, _: LightSampleContext, _: Vec3, _: bool) -> f32 {
        0.0
    }

    fn sample_radiance_emitted(&self, u1: Vec2, _: Vec2, lambda: &SampledWavelengths, time: f32) -> Option<LightRadianceEmittedSample> {
        let point = self.render_from_light.transform_point3(Vec3::ZERO);
        let ray = Ray::new(point, sampling::sample_uniform_sphere(u1), time, self.medium_interface.outside);

        Some(LightRadianceEmittedSample {
            radiance: self.intensity.sample(lambda) * self.scale,
            interaction: None,
            ray,
            pdf_position: 1.0,
            pdf_direction: sampling::uniform_sphere_pdf(),
        })
    }

    fn pdf_radiance_emitted(&self, _: Ray) -> (f32, f32) {
        (0.0, sampling::uniform_sphere_pdf())
    }

    fn radiance(&self, _: Vec3, _: Vec3, _: Vec2, _: Vec3, _: &SampledWavelengths) -> SampledSpectrum {
        SampledSpectrum::trivial()
    }

    fn radiance_emitted(&self, _: Ray, _: &SampledWavelengths) -> SampledSpectrum {
        SampledSpectrum::trivial()
    }

    fn preprocess(&mut self, _: Vec3, _: Vec3) {}
}
