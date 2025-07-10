use core::f32::consts;
use spirv_std::num_traits::Float;
use spirv_std::glam::{Vec2, Vec3};

#[inline]
pub fn sample_uniform_disk_polar(u: Vec2) -> Vec2 {
    let r = u.x.sqrt();
    let theta = consts::TAU * u.y;
    Vec2::new(r * theta.cos(), r * theta.sin())
}

#[inline]
pub fn sample_uniform_disk_concentric(u: Vec2) -> Vec2 {
    let uo = u * 2.0 - Vec2::new(1.0, 1.0);

    if uo.x == 0.0 || uo.y == 0.0 {
        Vec2::ZERO
    } else {
        let (r, theta) = if uo.x.abs() > uo.y.abs() {
            (uo.x, consts::FRAC_PI_4 * uo.y / uo.x)
        } else {
            (uo.y, consts::FRAC_PI_2 - consts::FRAC_PI_4 * uo.x / uo.y)
        };

        r * Vec2::new(theta.cos(), theta.sin())
    }
}

#[inline]
pub fn sample_uniform_sphere(u: Vec2) -> Vec3 {
    let z = 1.0 - 2.0 * u.x;
    let r = (1.0 - z * z).sqrt();
    let phi = consts::TAU * u.y;

    Vec3::new(r * phi.cos(), r * phi.sin(), z)
}

#[inline]
pub fn sample_uniform_hemisphere_concentric(u: Vec2) -> Vec3 {
    let uo = u * 2.0 - Vec2::new(1.0, 1.0);

    if uo.x == 0.0 || uo.y == 0.0 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        let (r, theta) = if uo.x.abs() > uo.y.abs() {
            (uo.x, consts::FRAC_PI_4 * uo.y / uo.x)
        } else {
            (uo.y, consts::FRAC_PI_2 - consts::FRAC_PI_4 * uo.x / uo.y)
        };

        Vec3::new(theta.cos() * r * (2.0 - r * r).sqrt(), theta.sin() * r * (2.0 - r * r).sqrt(), 1.0 - r * r)
    }
}

#[inline]
pub fn sample_uniform_hemisphere(u: Vec2) -> Vec3 {
    let z = u.x;
    let r = (1.0 - z * z).sqrt();
    let phi = consts::TAU * u.y;

    Vec3::new(r * phi.cos(), r * phi.sin(), z)
}

#[inline]
pub fn sample_cosine_hemisphere(u: Vec2) -> Vec3 {
    let d = sample_uniform_disk_concentric(u);
    let z = (1.0 - d.x * d.x - d.y * d.y).sqrt();

    Vec3::new(d.x, d.y, z)
}

#[inline]
pub fn uniform_sphere_pdf() -> f32 {
    consts::FRAC_1_PI * 0.25
}

#[inline]
pub fn uniform_hemisphere_pdf() -> f32 {
    consts::FRAC_1_PI * 0.5
}

#[inline]
pub fn cosine_hemisphere_pdf(cos_theta: f32) -> f32 {
    cos_theta * consts::FRAC_1_PI
}
