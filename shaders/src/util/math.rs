use core::ops::{Add, Mul};
use num_traits::Float;
use spirv_std::glam::Mat4;

pub fn sqr<T>(val: T) -> T
where
    T: Copy + Mul<Output = T>,
{
    val * val
}

pub fn lerp<T>(t: f32, a: T, b: T) -> T
where
    T: Add<Output = T> + Mul<f32, Output = T>,
{
    a * (1.0 - t) + b * t
}

pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

pub fn perspective(fov: f32, near: f32, far: f32) -> Mat4 {
    let inv_tan = 1.0 / (0.5 * fov).tan();
    let f_n = far / (far - near);

    Mat4::from_cols_array_2d(&[[inv_tan, 0.0, 0.0, 0.0], [0.0, inv_tan, 0.0, 0.0], [0.0, 0.0, f_n, 1.0], [0.0, 0.0, -near * f_n, 0.0]])
}

#[inline]
pub fn powi(x: f32, n: i32) -> f32 {
    match n {
        0 => 1.0,
        1 => x,
        2 => x * x,
        3 => x * x * x,
        4 => {
            let x2 = x * x;
            x2 * x2
        }
        _ => {
            let mut result = 1.0;
            let mut base = x;
            let mut exp = n.abs() as u32;

            while exp > 0 {
                if exp % 2 == 1 {
                    result *= base;
                }
                base *= base;
                exp /= 2;
            }

            if n < 0 {
                1.0 / result
            } else {
                result
            }
        }
    }
}

#[macro_export]
macro_rules! calc_polynomial {
    ($t:expr, $last:expr) => { $last };
    ($t:expr, $first:expr, $($rest:expr),+) => {
        $t.mul_add(calc_polynomial!($t, $($rest),+), $first)
    };
}
