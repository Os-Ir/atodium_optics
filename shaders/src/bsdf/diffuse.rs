use crate::bsdf::{Bsdf, BsdfFlags, BsdfReflTransFlags, BsdfSample, TransportMode};
use crate::spectrum::SampledSpectrum;
use crate::util::sampling;
use core::f32::consts;
use spirv_std::glam::{Vec2, Vec3};
use spirv_std::num_traits::Float;

#[repr(C)]
pub struct DiffuseBsdf {
    reflect: SampledSpectrum,
}

impl Bsdf for DiffuseBsdf {
    fn flags(&self) -> BsdfFlags {
        if self.reflect.is_nontrivial() {
            BsdfFlags::DIFFUSE_REFLECTION
        } else {
            BsdfFlags::UNSET
        }
    }

    fn bsdf_func(&self, output_direction: Vec3, input_direction: Vec3, _: TransportMode) -> SampledSpectrum {
        if input_direction.z * output_direction.z > 0.0 {
            self.reflect * consts::FRAC_1_PI
        } else {
            SampledSpectrum::trivial()
        }
    }

    fn sample(&self, output_direction: Vec3, _: f32, u: Vec2, _: TransportMode, sample_flags: BsdfReflTransFlags) -> Option<BsdfSample> {
        if sample_flags.contains(BsdfReflTransFlags::REFLECTION) {
            let mut input_direction = sampling::sample_cosine_hemisphere(u);

            if output_direction.z < 0.0 {
                input_direction.z *= -1.0;
            }

            let pdf = sampling::cosine_hemisphere_pdf(input_direction.z.abs());

            Some(BsdfSample {
                sampled_func: self.reflect * consts::FRAC_1_PI,
                input_direction,
                pdf,
                flags: BsdfFlags::DIFFUSE_REFLECTION,
                eta: 1.0,
                pdf_is_proportional: false,
            })
        } else {
            None
        }
    }

    fn pdf(&self, output_direction: Vec3, input_direction: Vec3, _: TransportMode, sample_flags: BsdfReflTransFlags) -> f32 {
        if sample_flags.contains(BsdfReflTransFlags::REFLECTION) && input_direction.z * output_direction.z > 0.0 {
            sampling::cosine_hemisphere_pdf(input_direction.z.abs())
        } else {
            0.0
        }
    }

    fn regularize(&mut self) {}
}
