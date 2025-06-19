use crate::util::vector::BasicVecOperation;
use spirv_std::glam::{UVec2, UVec3, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use spirv_std::num_traits::Float;
use spirv_std::ray_tracing::{AccelerationStructure, RayFlags};
use spirv_std::{spirv, Image};

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Payload {
    rand_state: u32,
    color: Vec3,
    ray_origin: Vec3,
    ray_direction: Vec3,
    ray_hit_sky: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct HitResult {
    position: Vec3,
    normal: Vec3,
    color: Vec3,
}

fn gen_rand(rand_state: &mut u32) -> f32 {
    *rand_state = (*rand_state) * 747796405 + 1;

    let r = *rand_state;
    let mut word = ((r >> ((r >> 28_u32) + 4)) ^ r) * 277803737;
    word = (word >> 22) ^ word;

    word as f32 / 4294967295.0
}

fn sky_color(direction: Vec3) -> Vec3 {
    if direction.y > 0.0 {
        Vec3::new(1.0, 1.0, 1.0).lerp(Vec3::new(0.25, 0.5, 1.0), direction.y)
    } else {
        Vec3::new(0.03, 0.03, 0.03)
    }
}

fn get_hit_result(primitive_id: usize, vertices: &[Vec4], indices: &[u32], hit_uv: Vec2, ray_direction: Vec3) -> HitResult {
    let i0 = indices[3 * primitive_id + 0] as usize;
    let i1 = indices[3 * primitive_id + 1] as usize;
    let i2 = indices[3 * primitive_id + 2] as usize;

    let v0 = vertices[i0].xyz();
    let v1 = vertices[i1].xyz();
    let v2 = vertices[i2].xyz();

    let barycentrics: Vec3 = Vec3::new(1.0 - hit_uv.x - hit_uv.y, hit_uv.x, hit_uv.y);

    let normal = (v1 - v0).cross(v2 - v0).normalize();

    HitResult {
        position: v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z,
        normal: normal.faceforward(ray_direction),
        color: Vec3::new(0.7, 0.7, 0.7),
    }
}

#[spirv(ray_generation)]
pub fn main_rgen(
    #[spirv(ray_payload)] payload: &mut Payload,

    #[spirv(launch_id)] launch_id: UVec3,

    #[spirv(descriptor_set = 0, binding = 0)] tlas: &AccelerationStructure,
    #[spirv(descriptor_set = 0, binding = 1)] image_output: &Image!(2D, format = rgba32f, sampled = false),
) {
    let resolution = UVec2::new(800, 600);
    let pixel = launch_id.xy();

    if pixel.x > resolution.x && pixel.y > resolution.y {
        return;
    }

    let camera_origin = Vec3::new(-0.001, 1.0, 6.0);
    let fov_vertical_slope: f32 = 1.0 / 5.0;
    let sample_level: u32 = 1024;
    let reflect_level: u32 = 32;
    let t_min: f32 = 0.0;
    let t_max: f32 = 10000.0;

    payload.rand_state = resolution.x * pixel.y + pixel.x;
    let mut integrated_color = Vec3::ZERO;

    for _ in 0..sample_level {
        let pixel_center: Vec2 = pixel.as_vec2() + Vec2::new(gen_rand(&mut payload.rand_state), gen_rand(&mut payload.rand_state));
        let screen_uv: Vec2 = Vec2::new(
            (2.0 * pixel_center.x - resolution.x as f32) / resolution.y as f32,
            -(2.0 * pixel_center.y - resolution.y as f32) / resolution.y as f32,
        );

        let mut ray_origin = camera_origin;
        let mut ray_direction = Vec3::new(fov_vertical_slope * screen_uv.x, fov_vertical_slope * screen_uv.y, -1.0).normalize();

        let mut current_ray_color = Vec3::new(1.0, 1.0, 1.0);

        for _ in 0..reflect_level {
            unsafe { tlas.trace_ray(RayFlags::OPAQUE, 0xff, 0, 0, 0, ray_origin, t_min, ray_direction, t_max, payload) };

            current_ray_color *= payload.color;

            if payload.ray_hit_sky > 0 {
                integrated_color += current_ray_color;
                break;
            } else {
                ray_origin = payload.ray_origin;
                ray_direction = payload.ray_direction;
            }
        }
    }

    integrated_color = integrated_color / sample_level as f32;

    unsafe { image_output.write(pixel, Vec4::new(integrated_color.x, integrated_color.y, integrated_color.z, 1.0)) };
}

#[spirv(closest_hit)]
pub fn main_rchit(
    #[spirv(incoming_ray_payload)] payload: &mut Payload,

    #[spirv(world_ray_direction)] world_ray_direction: Vec3,
    #[spirv(primitive_id)] primitive_id: u32,
    #[spirv(hit_attribute)] hit_attributes: &Vec2,

    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] vertices: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] indices: &[u32],
) {
    let hit_result = get_hit_result(primitive_id as _, vertices, indices, *hit_attributes, world_ray_direction);

    let phi = 6.2831853 * gen_rand(&mut payload.rand_state);
    let u = 2.0 * gen_rand(&mut payload.rand_state) - 1.0;
    let r = Float::sqrt(1.0 - u * u);

    payload.color = hit_result.color;
    payload.ray_origin = hit_result.position + 0.0001 * hit_result.normal;
    payload.ray_direction = (hit_result.normal + Vec3::new(r * Float::cos(phi), r * Float::sin(phi), u)).normalize();
    payload.ray_hit_sky = 0;
}

#[spirv(miss)]
pub fn main_rmiss(#[spirv(incoming_ray_payload)] payload: &mut Payload, #[spirv(world_ray_direction)] world_ray_direction: Vec3) {
    payload.color = sky_color(world_ray_direction);
    payload.ray_hit_sky = 1;
}
