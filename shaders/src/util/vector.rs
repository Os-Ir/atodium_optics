use num_traits::Float;
use spirv_std::glam::{Vec3, Vec3A};

pub trait BasicVecOperation: Copy {
    fn reflect(&self, normal: Self) -> Self;

    fn faceforward(&self, reference_normal: Self) -> Self;

    fn cos_theta(&self) -> f32;

    fn sin_theta(&self) -> f32;

    fn tan_theta(&self) -> f32;

    fn cos_theta_sqr(&self) -> f32;

    fn sin_theta_sqr(&self) -> f32;

    fn tan_theta_sqr(&self) -> f32;

    fn sin_phi(&self) -> f32;

    fn cos_phi(&self) -> f32;
}

impl BasicVecOperation for Vec3 {
    #[inline]
    fn reflect(&self, normal: Self) -> Self {
        *self - 2.0 * normal.dot(*self) * normal
    }

    #[inline]
    fn faceforward(&self, reference_normal: Self) -> Self {
        if reference_normal.dot(*self) < 0.0 {
            *self
        } else {
            -*self
        }
    }

    #[inline]
    fn cos_theta(&self) -> f32 {
        self.z
    }

    #[inline]
    fn sin_theta(&self) -> f32 {
        Float::sqrt(self.sin_theta_sqr())
    }

    #[inline]
    fn tan_theta(&self) -> f32 {
        self.sin_theta() / self.cos_theta()
    }

    #[inline]
    fn cos_theta_sqr(&self) -> f32 {
        self.z * self.z
    }

    #[inline]
    fn sin_theta_sqr(&self) -> f32 {
        Float::max(1.0 - self.cos_theta_sqr(), 0.0)
    }

    #[inline]
    fn tan_theta_sqr(&self) -> f32 {
        self.sin_theta_sqr() / self.cos_theta_sqr()
    }

    #[inline]
    fn sin_phi(&self) -> f32 {
        let sin_theta = self.sin_theta();

        if sin_theta == 0.0 {
            1.0
        } else {
            Float::clamp(self.y / sin_theta, -1.0, 1.0)
        }
    }

    #[inline]
    fn cos_phi(&self) -> f32 {
        let sin_theta = self.sin_theta();

        if sin_theta == 0.0 {
            0.0
        } else {
            Float::clamp(self.x / sin_theta, -1.0, 1.0)
        }
    }
}

impl BasicVecOperation for Vec3A {
    #[inline]
    fn reflect(&self, normal: Self) -> Self {
        *self - 2.0 * normal.dot(*self) * normal
    }

    #[inline]
    fn faceforward(&self, reference_normal: Self) -> Self {
        if reference_normal.dot(*self) < 0.0 {
            *self
        } else {
            -*self
        }
    }

    #[inline]
    fn cos_theta(&self) -> f32 {
        self.z
    }

    #[inline]
    fn sin_theta(&self) -> f32 {
        Float::sqrt(self.sin_theta_sqr())
    }

    #[inline]
    fn tan_theta(&self) -> f32 {
        self.sin_theta() / self.cos_theta()
    }

    #[inline]
    fn cos_theta_sqr(&self) -> f32 {
        self.z * self.z
    }

    #[inline]
    fn sin_theta_sqr(&self) -> f32 {
        Float::max(1.0 - self.cos_theta_sqr(), 0.0)
    }

    #[inline]
    fn tan_theta_sqr(&self) -> f32 {
        self.sin_theta_sqr() / self.cos_theta_sqr()
    }

    #[inline]
    fn sin_phi(&self) -> f32 {
        let sin_theta = self.sin_theta();

        if sin_theta == 0.0 {
            1.0
        } else {
            Float::clamp(self.y / sin_theta, -1.0, 1.0)
        }
    }

    #[inline]
    fn cos_phi(&self) -> f32 {
        let sin_theta = self.sin_theta();

        if sin_theta == 0.0 {
            0.0
        } else {
            Float::clamp(self.x / sin_theta, -1.0, 1.0)
        }
    }
}
