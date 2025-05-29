use spirv_std::glam::{UVec2, UVec3, Vec3Swizzles, Vec4};
use spirv_std::spirv;

const RESOLUTION: UVec2 = UVec2::new(800, 600);

#[spirv(compute(threads(16, 8, 1)))]
pub fn main_cs(#[spirv(global_invocation_id)] invocation_id: UVec3, #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] storage_image: &mut [Vec4]) {
    let pixel = invocation_id.xy();

    if pixel.x > RESOLUTION.x && pixel.y > RESOLUTION.y {
        return;
    }

    let pixel_color = Vec4::new((pixel.x as f32) / (RESOLUTION.x as f32), (pixel.y as f32) / (RESOLUTION.y as f32), 0.5, 1.0);

    let linear_idx = RESOLUTION.x * pixel.y + pixel.x;

    storage_image[linear_idx as usize] = pixel_color;
}
