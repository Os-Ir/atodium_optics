pub mod frame;
pub mod math;
pub mod sampling;
pub mod vector;

pub fn find_interval(sz: usize, pred: impl Fn(usize) -> bool) -> usize {
    if sz <= 1 {
        return 0;
    }

    let mut first = 1;
    let mut size = sz - 2;

    while size > 0 {
        let half = size >> 1;
        let mid = first + half;

        if pred(mid) {
            first = mid + 1;
            size -= half + 1;
        } else {
            size = half;
        }
    }

    let result = first - 1;

    if result >= sz {
        sz - 2
    } else if result == 0 {
        0
    } else {
        result
    }
}

#[inline]
pub fn next_float_up(x: f32) -> f32 {
    if x.is_infinite() && x > 0.0 {
        return x;
    }
    if x == -0.0 {
        return f32::MIN_POSITIVE;
    }
    let bits = x.to_bits();
    let next_bits = if x >= 0.0 { bits + 1 } else { bits - 1 };
    f32::from_bits(next_bits)
}

#[inline]
pub fn next_float_down(x: f32) -> f32 {
    if x.is_infinite() && x < 0.0 {
        return x;
    }
    if x == 0.0 {
        return -f32::MIN_POSITIVE;
    }
    let bits = x.to_bits();
    let next_bits = if x > 0.0 { bits - 1 } else { bits + 1 };
    f32::from_bits(next_bits)
}

#[inline(always)]
pub const fn cast_slice<T: Copy + Default, const N: usize, const M: usize>(input: [T; N], default_val: T) -> [T; M] {
    let mut result = [default_val; M];
    let mut i = 0;

    while i < N && i < M {
        result[i] = input[i];
        i += 1;
    }

    result
}
