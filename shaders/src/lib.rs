#![no_std]
#![feature(asm_experimental_arch, asm_const)]

use spirv_std::glam::{Vec3, Vec3A};
use spirv_std::num_traits::Float;

pub mod test;

pub trait BasicVecOperation {
    fn reflect(&self, normal: &Self) -> Self;

    fn refract(&self, normal: &Self, eta: f32) -> Self;

    fn faceforward(&self, reference_normal: &Self) -> Self;
}

impl BasicVecOperation for Vec3 {
    #[inline]
    fn reflect(&self, normal: &Self) -> Self {
        (*self) - 2.0 * normal.dot(*self) * (*normal)
    }

    #[inline]
    fn refract(&self, normal: &Self, eta: f32) -> Self {
        let dot_ni = normal.dot(*self);
        let k = 1.0 - eta * eta * (1.0 - dot_ni * dot_ni);
        if k < 0.0 {
            Vec3::ZERO
        } else {
            (*self) * eta - (*normal) * (eta * dot_ni + Float::sqrt(k))
        }
    }

    #[inline]
    fn faceforward(&self, reference_normal: &Self) -> Self {
        if reference_normal.dot(*self) < 0.0 {
            *self
        } else {
            -(*self)
        }
    }
}

impl BasicVecOperation for Vec3A {
    #[inline]
    fn reflect(&self, normal: &Self) -> Self {
        (*self) - 2.0 * normal.dot(*self) * (*normal)
    }

    #[inline]
    fn refract(&self, normal: &Self, eta: f32) -> Self {
        let dot_ni = normal.dot(*self);
        let k = 1.0 - eta * eta * (1.0 - dot_ni * dot_ni);
        if k < 0.0 {
            Vec3A::ZERO
        } else {
            (*self) * eta - (*normal) * (eta * dot_ni + Float::sqrt(k))
        }
    }

    #[inline]
    fn faceforward(&self, reference_normal: &Self) -> Self {
        if reference_normal.dot(*self) < 0.0 {
            *self
        } else {
            -(*self)
        }
    }
}
