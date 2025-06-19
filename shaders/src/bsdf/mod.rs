pub mod diffuse;
pub mod fresnel;

use crate::spectrum::SampledSpectrum;
use crate::util::sampling;
use bitflags::bitflags;
use core::f32::consts;
use spirv_std::glam::{Vec2, Vec3};
use spirv_std::num_traits::Float;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BsdfReflTransFlags: u32 {
        const UNSET                 = 0;
        const REFLECTION            = 1 << 0;
        const TRANSMISSION          = 1 << 1;

        const ALL                   = Self::REFLECTION.bits() | Self::TRANSMISSION.bits();
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BsdfFlags: u32 {
        const UNSET                 = 0;
        const REFLECTION            = 1 << 0;
        const TRANSMISSION          = 1 << 1;
        const DIFFUSE               = 1 << 2;
        const GLOSSY                = 1 << 3;
        const SPECULAR              = 1 << 4;

        const DIFFUSE_REFLECTION    = Self::DIFFUSE .bits() | Self::REFLECTION  .bits();
        const DIFFUSE_TRANSMISSION  = Self::DIFFUSE .bits() | Self::TRANSMISSION.bits();
        const GLOSSY_REFLECTION     = Self::GLOSSY  .bits() | Self::REFLECTION  .bits();
        const GLOSSY_TRANSMISSION   = Self::GLOSSY  .bits() | Self::TRANSMISSION.bits();
        const SPECULAR_REFLECTION   = Self::SPECULAR.bits() | Self::REFLECTION  .bits();
        const SPECULAR_TRANSMISSION = Self::SPECULAR.bits() | Self::TRANSMISSION.bits();
        const ALL                   = Self::DIFFUSE .bits() | Self::GLOSSY      .bits() | Self::SPECULAR.bits() | Self::REFLECTION.bits() | Self::TRANSMISSION.bits();
    }
}

pub enum TransportMode {
    Radiance,
    Importance,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct BsdfSample {
    sampled_func: SampledSpectrum,
    input_direction: Vec3,
    pdf: f32,
    flags: BsdfFlags,
    eta: f32,
    pdf_is_proportional: bool,
}

pub trait Bsdf {
    fn flags(&self) -> BsdfFlags;

    fn bsdf_func(&self, output_direction: Vec3, input_direction: Vec3, mode: TransportMode) -> SampledSpectrum;

    fn sample(&self, output_direction: Vec3, uc: f32, u: Vec2, mode: TransportMode, sample_flags: BsdfReflTransFlags) -> Option<BsdfSample>;

    fn pdf(&self, output_direction: Vec3, input_direction: Vec3, mode: TransportMode, sample_flags: BsdfReflTransFlags) -> f32;

    fn regularize(&mut self);

    fn single_sampled_reflectance(&self, output_direction: Vec3, uc: &[f32], u: &[Vec2]) -> SampledSpectrum {
        if output_direction.z == 0.0 {
            SampledSpectrum::trivial()
        } else {
            assert_eq!(uc.len(), u.len());

            let mut result = SampledSpectrum::trivial();

            for i in 0..uc.len() {
                if let Some(sample) = self.sample(output_direction, uc[i], u[i], TransportMode::Radiance, BsdfReflTransFlags::ALL) {
                    if sample.pdf > 0.0 {
                        result += sample.sampled_func * sample.input_direction.z.abs() / sample.pdf;
                    }
                }
            }

            result / uc.len() as f32
        }
    }

    fn dual_sampled_reflectance(&self, uc: &[f32], u1: &[Vec2], u2: &[Vec2]) -> SampledSpectrum {
        assert_eq!(uc.len(), u1.len());
        assert_eq!(uc.len(), u2.len());

        let mut result = SampledSpectrum::trivial();

        for i in 0..uc.len() {
            let output_direction = sampling::sample_uniform_hemisphere(u1[i]);

            if output_direction.z == 0.0 {
                continue;
            }

            let output_pdf = sampling::uniform_hemisphere_pdf();

            if let Some(sample) = self.sample(output_direction, uc[i], u2[i], TransportMode::Radiance, BsdfReflTransFlags::ALL) {
                if sample.pdf > 0.0 {
                    result += sample.sampled_func * sample.input_direction.z.abs() * output_direction.z.abs() / output_pdf / sample.pdf;
                }
            }
        }

        result / uc.len() as f32 / consts::PI
    }
}
