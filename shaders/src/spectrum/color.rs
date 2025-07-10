use crate::util::math;
use crate::{calc_polynomial, util};
use core::array;
use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};
use spirv_std::num_traits::Float;
use spirv_std::glam::{Mat3, Vec3};

#[rustfmt::skip]
pub const SRGB_TO_LINEAR_LUT: [f32; 256] = [
    0.0000000000, 0.0003035270, 0.0006070540, 0.0009105810, 0.0012141080, 0.0015176350, 0.0018211619, 0.0021246888,
    0.0024282159, 0.0027317430, 0.0030352699, 0.0033465356, 0.0036765069, 0.0040247170, 0.0043914421, 0.0047769533,
    0.0051815170, 0.0056053917, 0.0060488326, 0.0065120910, 0.0069954102, 0.0074990317, 0.0080231922, 0.0085681248,
    0.0091340570, 0.0097212177, 0.0103298230, 0.0109600937, 0.0116122449, 0.0122864870, 0.0129830306, 0.0137020806,
    0.0144438436, 0.0152085144, 0.0159962922, 0.0168073755, 0.0176419523, 0.0185002182, 0.0193823613, 0.0202885624,
    0.0212190095, 0.0221738834, 0.0231533647, 0.0241576303, 0.0251868572, 0.0262412224, 0.0273208916, 0.0284260381,
    0.0295568332, 0.0307134409, 0.0318960287, 0.0331047624, 0.0343398079, 0.0356013142, 0.0368894450, 0.0382043645,
    0.0395462364, 0.0409151986, 0.0423114114, 0.0437350273, 0.0451862030, 0.0466650836, 0.0481718220, 0.0497065634,
    0.0512694679, 0.0528606549, 0.0544802807, 0.0561284944, 0.0578054339, 0.0595112406, 0.0612460710, 0.0630100295,
    0.0648032799, 0.0666259527, 0.0684781820, 0.0703601092, 0.0722718611, 0.0742135793, 0.0761853904, 0.0781874284,
    0.0802198276, 0.0822827145, 0.0843762159, 0.0865004659, 0.0886556059, 0.0908417329, 0.0930589810, 0.0953074843,
    0.0975873619, 0.0998987406, 0.1022417471, 0.1046164930, 0.1070231125, 0.1094617173, 0.1119324341, 0.1144353822,
    0.1169706732, 0.1195384338, 0.1221387982, 0.1247718409, 0.1274376959, 0.1301364899, 0.1328683347, 0.1356333494,
    0.1384316236, 0.1412633061, 0.1441284865, 0.1470272839, 0.1499598026, 0.1529261619, 0.1559264660, 0.1589608639,
    0.1620294005, 0.1651322246, 0.1682693958, 0.1714410931, 0.1746473908, 0.1778884083, 0.1811642349, 0.1844749898,
    0.1878207624, 0.1912016720, 0.1946178079, 0.1980693042, 0.2015562356, 0.2050787061, 0.2086368501, 0.2122307271,
    0.2158605307, 0.2195262313, 0.2232279778, 0.2269658893, 0.2307400703, 0.2345506549, 0.2383976579, 0.2422811985,
    0.2462013960, 0.2501583695, 0.2541521788, 0.2581829131, 0.2622507215, 0.2663556635, 0.2704978585, 0.2746773660,
    0.2788943350, 0.2831487954, 0.2874408960, 0.2917706966, 0.2961383164, 0.3005438447, 0.3049873710, 0.3094689548,
    0.3139887452, 0.3185468316, 0.3231432438, 0.3277781308, 0.3324515820, 0.3371636569, 0.3419144452, 0.3467040956,
    0.3515326977, 0.3564002514, 0.3613068759, 0.3662526906, 0.3712377846, 0.3762622178, 0.3813261092, 0.3864295185,
    0.3915725648, 0.3967553079, 0.4019778669, 0.4072403014, 0.4125427008, 0.4178851545, 0.4232677519, 0.4286905527,
    0.4341537058, 0.4396572411, 0.4452012479, 0.4507858455, 0.4564110637, 0.4620770514, 0.4677838385, 0.4735315442,
    0.4793202281, 0.4851499796, 0.4910208881, 0.4969330430, 0.5028865933, 0.5088814497, 0.5149177909, 0.5209956765,
    0.5271152258, 0.5332764983, 0.5394796133, 0.5457245708, 0.5520114899, 0.5583404899, 0.5647116303, 0.5711249113,
    0.5775805116, 0.5840784907, 0.5906189084, 0.5972018838, 0.6038274169, 0.6104956269, 0.6172066331, 0.6239604354,
    0.6307572126, 0.6375969648, 0.6444797516, 0.6514056921, 0.6583748460, 0.6653873324, 0.6724432111, 0.6795425415,
    0.6866854429, 0.6938719153, 0.7011020184, 0.7083759308, 0.7156936526, 0.7230552435, 0.7304608822, 0.7379105687,
    0.7454043627, 0.7529423237, 0.7605246305, 0.7681512833, 0.7758223414, 0.7835379243, 0.7912980318, 0.7991028428,
    0.8069523573, 0.8148466945, 0.8227858543, 0.8307699561, 0.8387991190, 0.8468732834, 0.8549926877, 0.8631572723,
    0.8713672161, 0.8796223402, 0.8879231811, 0.8962693810, 0.9046613574, 0.9130986929, 0.9215820432, 0.9301108718,
    0.9386858940, 0.9473065734, 0.9559735060, 0.9646862745, 0.9734454751, 0.9822505713, 0.9911022186, 1.0000000000,
];

const LMS_FROM_XYZ: Mat3 = Mat3::from_cols_array(&[0.8951, -0.7502, 0.0389, 0.2664, 1.7135, -0.0685, -0.1614, 0.0367, 1.0296]);

const XYZ_FROM_LMS: Mat3 = Mat3::from_cols_array(&[0.986993, 0.432305, -0.00852866, -0.147054, 0.51836, 0.0400428, 0.159963, 0.0492912, 0.968487]);

#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct RgbColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl RgbColor {
    pub unsafe fn new_unchecked(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    pub fn new(mut r: f32, mut g: f32, mut b: f32) -> Self {
        r = r.clamp(0.0, 1.0);
        g = g.clamp(0.0, 1.0);
        b = b.clamp(0.0, 1.0);

        unsafe { Self::new_unchecked(r, g, b) }
    }

    pub fn average(&self) -> f32 {
        (self.r + self.g + self.b) / 3.0
    }

    pub fn clamp(&self, mut min: f32, mut max: f32) -> Self {
        min = min.clamp(0.0, 1.0);
        max = max.clamp(min, 1.0);

        Self {
            r: self.r.clamp(min, max),
            g: self.g.clamp(min, max),
            b: self.b.clamp(min, max),
        }
    }
}

impl Into<[f32; 3]> for RgbColor {
    fn into(self) -> [f32; 3] {
        [self.r, self.g, self.b]
    }
}

impl Into<Vec3> for RgbColor {
    fn into(self) -> Vec3 {
        Vec3::new(self.r, self.g, self.b)
    }
}

impl Index<usize> for RgbColor {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.r,
            1 => &self.g,
            2 => &self.b,
            _ => panic!("Index out of bounds for RgbColor"),
        }
    }
}

impl IndexMut<usize> for RgbColor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.r,
            1 => &mut self.g,
            2 => &mut self.b,
            _ => panic!("Index out of bounds for RgbColor"),
        }
    }
}

impl Add for RgbColor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.r + rhs.r, self.g + rhs.g, self.b + rhs.b)
    }
}

impl AddAssign for RgbColor {
    fn add_assign(&mut self, rhs: Self) {
        self.r = (self.r + rhs.r).clamp(0.0, 1.0);
        self.g = (self.g + rhs.g).clamp(0.0, 1.0);
        self.b = (self.b + rhs.b).clamp(0.0, 1.0);
    }
}

impl Sub for RgbColor {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.r - rhs.r, self.g - rhs.g, self.b - rhs.b)
    }
}

impl SubAssign for RgbColor {
    fn sub_assign(&mut self, rhs: Self) {
        self.r = (self.r - rhs.r).clamp(0.0, 1.0);
        self.g = (self.g - rhs.g).clamp(0.0, 1.0);
        self.b = (self.b - rhs.b).clamp(0.0, 1.0);
    }
}

impl Mul<f32> for RgbColor {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        Self::new(self.r * rhs, self.g * rhs, self.b * rhs)
    }
}

impl Mul<RgbColor> for f32 {
    type Output = RgbColor;

    fn mul(self, rhs: RgbColor) -> RgbColor {
        rhs * self
    }
}

impl MulAssign<f32> for RgbColor {
    fn mul_assign(&mut self, rhs: f32) {
        self.r = (self.r * rhs).clamp(0.0, 1.0);
        self.g = (self.g * rhs).clamp(0.0, 1.0);
        self.b = (self.b * rhs).clamp(0.0, 1.0);
    }
}

impl Div<f32> for RgbColor {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        if rhs == 0.0 {
            return Self::new(0.0, 0.0, 0.0);
        }

        self * (1.0 / rhs)
    }
}

impl DivAssign<f32> for RgbColor {
    fn div_assign(&mut self, rhs: f32) {
        self.mul_assign(1.0 / rhs)
    }
}

#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct XyzColor {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl XyzColor {
    pub unsafe fn new_unchecked(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn new(mut x: f32, mut y: f32, mut z: f32) -> Self {
        x = x.max(0.0);
        y = y.max(0.0);
        z = z.max(0.0);

        unsafe { Self::new_unchecked(x, y, z) }
    }

    pub fn average(&self) -> f32 {
        (self.x + self.y + self.z) / 3.0
    }

    pub fn xy(&self) -> (f32, f32) {
        let sum = self.x + self.y + self.z;
        (self.x / sum, self.y / sum)
    }

    pub fn from_xyy(x: f32, y: f32, y_val: f32) -> Self {
        if y == 0.0 {
            return Self::new(0.0, 0.0, 0.0);
        }
        Self::new(x * y_val / y, y_val, (1.0 - x - y) * y_val / y)
    }
}

impl Into<[f32; 3]> for XyzColor {
    fn into(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

impl Into<Vec3> for XyzColor {
    fn into(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

impl Index<usize> for XyzColor {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index out of bounds for XyzColor"),
        }
    }
}

impl IndexMut<usize> for XyzColor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Index out of bounds for XyzColor"),
        }
    }
}

impl Add for XyzColor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl AddAssign for XyzColor {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub for XyzColor {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl SubAssign for XyzColor {
    fn sub_assign(&mut self, rhs: Self) {
        self.x = (self.x - rhs.x).max(0.0);
        self.y = (self.y - rhs.y).max(0.0);
        self.z = (self.z - rhs.z).max(0.0);
    }
}

impl Mul<f32> for XyzColor {
    type Output = Self;

    fn mul(self, mut rhs: f32) -> Self {
        rhs = rhs.max(0.0);

        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Mul<XyzColor> for f32 {
    type Output = XyzColor;

    fn mul(self, rhs: XyzColor) -> XyzColor {
        rhs * self
    }
}

impl MulAssign<f32> for XyzColor {
    fn mul_assign(&mut self, rhs: f32) {
        self.x = (self.x * rhs).max(0.0);
        self.y = (self.y * rhs).max(0.0);
        self.z = (self.z * rhs).max(0.0);
    }
}

impl Div<f32> for XyzColor {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        if rhs == 0.0 {
            return Self::new(0.0, 0.0, 0.0);
        }

        self * (1.0 / rhs)
    }
}

impl DivAssign<f32> for XyzColor {
    fn div_assign(&mut self, rhs: f32) {
        self.mul_assign(1.0 / rhs)
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct RgbSigmoidPolynomial {
    c2: f32,
    c1: f32,
    c0: f32,
}

impl RgbSigmoidPolynomial {
    pub fn new(c2: f32, c1: f32, c0: f32) -> Self {
        Self { c2, c1, c0 }
    }

    pub fn max_value(&self) -> f32 {
        math::clamp(-0.5 * self.c1 / self.c2, crate::spectrum::LAMBDA_MIN, crate::spectrum::LAMBDA_MAX)
    }

    pub fn get_value(&self, lambda: f32) -> f32 {
        Self::sigmoid(calc_polynomial!(lambda, self.c0, self.c1, self.c2))
    }

    #[inline]
    fn sigmoid(x: f32) -> f32 {
        if x.is_infinite() {
            return if x.is_sign_positive() { 1.0 } else { 0.0 };
        }

        0.5 * (1.0 + x / (1.0 + x * x).sqrt())
    }
}

const RBG_TO_SPECTRUM_TABLE_RES: usize = 64;
type RbgToSpectrumTableCoefficients = [[[[[f32; 3]; RBG_TO_SPECTRUM_TABLE_RES]; RBG_TO_SPECTRUM_TABLE_RES]; RBG_TO_SPECTRUM_TABLE_RES]; 3];

#[derive(Clone, Copy)]
#[repr(C)]
pub struct RgbToSpectrumTable {
    coefficients: RbgToSpectrumTableCoefficients,
    z_node: [f32; RBG_TO_SPECTRUM_TABLE_RES],
}

impl RgbToSpectrumTable {
    pub fn color_to_polynomial(&self, rgb: RgbColor) -> RgbSigmoidPolynomial {
        if rgb.r == rgb.g && rgb.g == rgb.b {
            return RgbSigmoidPolynomial::new(0.0, 0.0, (rgb.r - 0.5) / (rgb.r * (1.0 - rgb.r)).sqrt());
        }

        let max_idx: usize = if rgb[0] > rgb[1] {
            if rgb[0] > rgb[2] {
                0
            } else {
                2
            }
        } else {
            if rgb[1] > rgb[2] {
                1
            } else {
                2
            }
        };

        let z = rgb[max_idx];
        let x = rgb[(max_idx + 1) % 3] * (RBG_TO_SPECTRUM_TABLE_RES as f32 - 1.0) / z;
        let y = rgb[(max_idx + 2) % 3] * (RBG_TO_SPECTRUM_TABLE_RES as f32 - 1.0) / z;

        let xi = (x as usize).min(RBG_TO_SPECTRUM_TABLE_RES - 2);
        let yi = (y as usize).min(RBG_TO_SPECTRUM_TABLE_RES - 2);
        let zi = util::find_interval(RBG_TO_SPECTRUM_TABLE_RES, |i| self.z_node[i] < z);

        let dx = x - xi as f32;
        let dy = y - yi as f32;
        let dz = (z - self.z_node[zi]) / (self.z_node[zi + 1] - self.z_node[zi]);

        let c: [f32; 3] = array::from_fn(|i| {
            let co: &dyn Fn(usize, usize, usize) -> f32 = &|dx_in, dy_in, dz_in| self.coefficients[max_idx][zi + dz_in][yi + dy_in][xi + dx_in][i];

            math::lerp(
                dz,
                math::lerp(dy, math::lerp(dx, co(0, 0, 0), co(1, 0, 0)), math::lerp(dx, co(0, 1, 0), co(1, 1, 0))),
                math::lerp(dy, math::lerp(dx, co(0, 0, 1), co(1, 0, 1)), math::lerp(dx, co(0, 1, 1), co(1, 1, 1))),
            )
        });

        RgbSigmoidPolynomial::new(c[0], c[1], c[2])
    }
}

pub fn white_balance(src_white: (f32, f32), target_white: (f32, f32)) -> Mat3 {
    let src_xyz = XyzColor::from_xyy(src_white.0, src_white.1, 1.0);
    let dst_xyz = XyzColor::from_xyy(target_white.0, target_white.1, 1.0);

    let src_lms = LMS_FROM_XYZ * Vec3::new(src_xyz.x, src_xyz.y, src_xyz.z);
    let dst_lms = LMS_FROM_XYZ * Vec3::new(dst_xyz.x, dst_xyz.y, dst_xyz.z);

    let lms_correct = Mat3::from_diagonal(Vec3::new(dst_lms.x / src_lms.x, dst_lms.y / src_lms.y, dst_lms.z / src_lms.z));

    XYZ_FROM_LMS * lms_correct * LMS_FROM_XYZ
}
