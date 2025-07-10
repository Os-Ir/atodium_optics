use crate::bsdf::{Bsdf, BsdfFlags, BsdfReflTransFlags, BsdfSample, TransportMode};
use crate::spectrum::{SampledSpectrum, N_SAMPLES};
use crate::util::vector::BasicVecOperation;
use crate::util::{math, sampling};
use core::array;
use core::f32::consts;
use num_complex::Complex32;
use spirv_std::num_traits::Float;
use spirv_std::glam::{Vec2, Vec3};

#[inline]
pub fn refract(input_direction: Vec3, mut normal: Vec3, mut eta: f32) -> Option<(f32, Vec3)> {
    let mut cos_theta_i = normal.dot(input_direction);

    if cos_theta_i < 0.0 {
        eta = 1.0 / eta;
        cos_theta_i = -cos_theta_i;
        normal = -normal;
    }

    let sin_theta_i_sqr = (1.0 - cos_theta_i * cos_theta_i).max(0.0);
    let sin_theta_t_sqr = sin_theta_i_sqr / (eta * eta);

    if sin_theta_t_sqr >= 1.0 {
        None
    } else {
        let cos_theta_t = (1.0 - sin_theta_t_sqr).sqrt();

        Some((eta, -input_direction / eta + (cos_theta_i / eta - cos_theta_t) * normal))
    }
}

#[inline]
pub fn fresnel_real(mut cos_theta_i: f32, mut eta: f32) -> f32 {
    cos_theta_i = cos_theta_i.clamp(-1.0, 1.0);

    if cos_theta_i < 0.0 {
        eta = 1.0 / eta;
        cos_theta_i = -cos_theta_i;
    }

    let sin_theta_i_sqr = 1.0 - cos_theta_i * cos_theta_i;
    let sin_theta_t_sqr = sin_theta_i_sqr / (eta * eta);

    if sin_theta_t_sqr >= 1.0 {
        1.0
    } else {
        let cos_theta_t = (1.0 - sin_theta_t_sqr).sqrt();

        let r_parallel = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
        let r_perpendicular = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);

        (r_parallel * r_parallel + r_perpendicular * r_perpendicular) * 0.5
    }
}

#[inline]
pub fn fresnel_complex(mut cos_theta_i: f32, eta: Complex32) -> f32 {
    cos_theta_i = cos_theta_i.clamp(0.0, 1.0);

    let sin_theta_i_sqr = 1.0 - cos_theta_i * cos_theta_i;
    let sin_theta_t_sqr: Complex32 = (sin_theta_i_sqr / (eta * eta)).into();
    let cos_theta_t = (1.0 - sin_theta_t_sqr).sqrt();

    let r_parallel = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    let r_perpendicular = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);

    (r_parallel.norm() + r_perpendicular.norm()) * 0.5
}

#[inline]
pub fn fresnel_real_sampled(cos_theta_i: f32, eta: SampledSpectrum) -> SampledSpectrum {
    let values: [f32; N_SAMPLES] = array::from_fn(|i| fresnel_real(cos_theta_i, eta[i]));
    SampledSpectrum::from_array(values)
}

#[inline]
pub fn fresnel_complex_sampled(cos_theta_i: f32, eta_re: SampledSpectrum, eta_im: SampledSpectrum) -> SampledSpectrum {
    let values: [f32; N_SAMPLES] = array::from_fn(|i| fresnel_complex(cos_theta_i, Complex32::new(eta_re[i], eta_im[i])));
    SampledSpectrum::from_array(values)
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct TrowbridgeReitzDistribution {
    alpha_x: f32,
    alpha_y: f32,
}

impl TrowbridgeReitzDistribution {
    pub fn new(alpha_x: f32, alpha_y: f32) -> Self {
        Self {
            alpha_x: alpha_x.max(1.0e-4),
            alpha_y: alpha_y.max(1.0e-4),
        }
    }

    pub fn distribution(&self, sub_normal: Vec3) -> f32 {
        let tan_theta_sqr = sub_normal.tan_theta_sqr();

        if tan_theta_sqr.is_finite() {
            let cos_theta_4 = math::sqr(sub_normal.cos_theta_sqr());

            if cos_theta_4 < 1.0e-16 {
                0.0
            } else {
                let e = tan_theta_sqr * (math::sqr(sub_normal.cos_phi() / self.alpha_x) + math::sqr(sub_normal.sin_phi() / self.alpha_y));
                1.0 / (consts::PI * self.alpha_x * self.alpha_y * cos_theta_4 * math::sqr(1.0 + e))
            }
        } else {
            0.0
        }
    }

    pub fn effectively_smooth(&self) -> bool {
        self.alpha_x.max(self.alpha_y) < 1.0e-3
    }

    fn mask_lambda(&self, direction: Vec3) -> f32 {
        let tan_theta_sqr = direction.tan_theta_sqr();

        if tan_theta_sqr.is_finite() {
            let alpha_sqr = math::sqr(direction.cos_phi() * self.alpha_x) + math::sqr(direction.sin_phi() * self.alpha_y);
            ((1.0 + alpha_sqr * tan_theta_sqr).sqrt() - 1.0) * 0.5
        } else {
            0.0
        }
    }

    pub fn masking_func(&self, direction: Vec3) -> f32 {
        1.0 / (1.0 + self.mask_lambda(direction))
    }

    pub fn masking_shadowing_func(&self, output_direction: Vec3, input_direction: Vec3) -> f32 {
        1.0 / (1.0 + self.mask_lambda(output_direction) + self.mask_lambda(input_direction))
    }

    pub fn pdf(&self, output_direction: Vec3, sub_normal: Vec3) -> f32 {
        (self.masking_func(output_direction) * self.distribution(sub_normal) * output_direction.dot(sub_normal) / output_direction.cos_theta()).abs()
    }

    pub fn sample(&self, output_direction: Vec3, u: Vec2) -> Vec3 {
        let mut wh = Vec3::new(self.alpha_x * output_direction.x, self.alpha_y * output_direction.y, output_direction.z);
        if wh.z < 0.0 {
            wh = -wh;
        }

        let tangent_x = if wh.z < 0.99999 {
            Vec3::new(0.0, 0.0, 1.0).cross(wh).normalize()
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };
        let tangent_y = wh.cross(tangent_x);

        let mut point = sampling::sample_uniform_disk_polar(u);

        let h = (1.0 - math::sqr(point.x)).sqrt();
        point.y = math::lerp(h, point.y, (1.0 + wh.z) * 0.5);

        let point_z = (1.0 - point.length_squared()).max(0.0).sqrt();
        let nh = point.x * tangent_x + point.y * tangent_y + point_z * wh;

        Vec3::new(self.alpha_x * nh.x, self.alpha_y * nh.y, nh.z.max(1.0e-6)).normalize()
    }

    pub fn regularize(&mut self) {
        if self.alpha_x < 0.3 {
            self.alpha_x = (self.alpha_x * 2.0).clamp(0.1, 0.3);
        }
        if self.alpha_y < 0.3 {
            self.alpha_y = (self.alpha_y * 2.0).clamp(0.1, 0.3);
        }
    }
}

#[repr(C)]
pub struct ConductorBsdf {
    eta_re: SampledSpectrum,
    eta_im: SampledSpectrum,
    roughness: TrowbridgeReitzDistribution,
}

impl Bsdf for ConductorBsdf {
    fn flags(&self) -> BsdfFlags {
        if self.roughness.effectively_smooth() {
            BsdfFlags::SPECULAR_REFLECTION
        } else {
            BsdfFlags::GLOSSY_REFLECTION
        }
    }

    fn bsdf_func(&self, output_direction: Vec3, input_direction: Vec3, _: TransportMode) -> SampledSpectrum {
        if output_direction.z * input_direction.z > 0.0 && !self.roughness.effectively_smooth() {
            let cos_theta_o = output_direction.cos_theta().abs();
            let cos_theta_i = input_direction.cos_theta().abs();

            if cos_theta_o == 0.0 || cos_theta_i == 0.0 {
                SampledSpectrum::trivial()
            } else {
                let mut sub_normal = output_direction + input_direction;

                if sub_normal.length_squared() == 0.0 {
                    SampledSpectrum::trivial()
                } else {
                    sub_normal = sub_normal.normalize();
                    let fresnel = fresnel_complex_sampled(output_direction.dot(sub_normal).abs(), self.eta_re, self.eta_im);
                    fresnel * self.roughness.distribution(sub_normal) * self.roughness.masking_shadowing_func(output_direction, input_direction) / (4.0 * cos_theta_o * cos_theta_i)
                }
            }
        } else {
            SampledSpectrum::trivial()
        }
    }

    fn sample(&self, output_direction: Vec3, _: f32, u: Vec2, _: TransportMode, sample_flags: BsdfReflTransFlags) -> Option<BsdfSample> {
        if !sample_flags.contains(BsdfReflTransFlags::REFLECTION) {
            return None;
        }

        if self.roughness.effectively_smooth() {
            let input_direction = Vec3::new(-output_direction.x, -output_direction.y, output_direction.z);
            let fresnel = fresnel_complex_sampled(input_direction.cos_theta().abs(), self.eta_re, self.eta_im);

            return Some(BsdfSample {
                sampled_func: fresnel,
                input_direction,
                pdf: 1.0,
                flags: BsdfFlags::SPECULAR_REFLECTION,
                eta: 1.0,
                pdf_is_proportional: false,
            });
        }

        if output_direction.z == 0.0 {
            return None;
        }

        let sub_normal = self.roughness.sample(output_direction, u);
        let input_direction = output_direction.reflect(sub_normal);

        if input_direction.z * output_direction.z <= 0.0 {
            return None;
        }

        let pdf = self.roughness.pdf(output_direction, sub_normal) / (4.0 * output_direction.dot(sub_normal).abs());
        let cos_theta_o = output_direction.cos_theta().abs();
        let cos_theta_i = input_direction.cos_theta().abs();

        if cos_theta_o == 0.0 || cos_theta_i == 0.0 {
            None
        } else {
            let fresnel = fresnel_complex_sampled(output_direction.dot(sub_normal).abs(), self.eta_re, self.eta_im);
            let sampled_func = fresnel * self.roughness.distribution(sub_normal) * self.roughness.masking_shadowing_func(output_direction, input_direction) / (4.0 * cos_theta_o * cos_theta_i);

            Some(BsdfSample {
                sampled_func,
                input_direction,
                pdf,
                flags: BsdfFlags::GLOSSY_REFLECTION,
                eta: 1.0,
                pdf_is_proportional: false,
            })
        }
    }

    fn pdf(&self, output_direction: Vec3, input_direction: Vec3, _: TransportMode, sample_flags: BsdfReflTransFlags) -> f32 {
        if !self.roughness.effectively_smooth() && output_direction.z * input_direction.z > 0.0 && sample_flags.contains(BsdfReflTransFlags::REFLECTION) {
            let mut sub_normal = output_direction + input_direction;

            if sub_normal.length_squared() == 0.0 {
                0.0
            } else {
                sub_normal = sub_normal.normalize().faceforward(Vec3::new(0.0, 0.0, 1.0));
                self.roughness.pdf(output_direction, sub_normal) / (4.0 * output_direction.dot(sub_normal).abs())
            }
        } else {
            0.0
        }
    }

    fn regularize(&mut self) {
        self.roughness.regularize()
    }
}
