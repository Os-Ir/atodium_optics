use crate::spectrum::color::{RgbColor, XyzColor};
use crate::spectrum::{DenselySampledSpectrum, ISpectrum};
use spirv_std::glam::{Mat3, Vec2, Vec3};

#[derive(Clone, Copy)]
#[repr(C)]
pub struct RgbColorSpace {
    pub illuminant: DenselySampledSpectrum,
    xyz_from_rgb: Mat3,
    rgb_from_xyz: Mat3,
    r: Vec2,
    g: Vec2,
    b: Vec2,
    w: Vec2,
}

impl RgbColorSpace {
    pub fn new(illuminant: DenselySampledSpectrum, r: Vec2, g: Vec2, b: Vec2) -> Self {
        let w_xyz = illuminant.to_xyz_color();
        let w = Vec2::new(w_xyz.x, w_xyz.y);

        let r_xyz = XyzColor::from_xyy(r.x, r.y, 1.0);
        let g_xyz = XyzColor::from_xyy(g.x, g.y, 1.0);
        let b_xyz = XyzColor::from_xyy(b.x, b.y, 1.0);

        let rgb = Mat3::from_cols_array_2d(&[r_xyz.into(), g_xyz.into(), b_xyz.into()]);

        let c: Vec3 = rgb.inverse() * <XyzColor as Into<Vec3>>::into(w_xyz);

        let xyz_from_rgb = rgb * Mat3::from_diagonal(c.into());
        let rgb_from_xyz = xyz_from_rgb.inverse();

        Self {
            illuminant,
            xyz_from_rgb,
            rgb_from_xyz,
            r,
            g,
            b,
            w,
        }
    }

    pub fn to_xyz(&self, rgb: RgbColor) -> Vec3 {
        self.xyz_from_rgb * <RgbColor as Into<Vec3>>::into(rgb)
    }

    pub fn to_rgb(&self, xyz: XyzColor) -> Vec3 {
        self.rgb_from_xyz * <XyzColor as Into<Vec3>>::into(xyz)
    }
}
