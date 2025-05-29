#version 460
#extension GL_EXT_ray_tracing: require
#extension GL_GOOGLE_include_directive: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_scalar_block_layout: enable

#include "rt.h.glsl"

layout (location = 0) rayPayloadInEXT PayloadInfo payload;

hitAttributeEXT vec2 hit_uv;

void main() {
    const vec3 ray_direction = gl_WorldRayDirectionEXT;

    HitResult hit_result = get_hit_result(gl_PrimitiveID, hit_uv, ray_direction);

    const float phi = 6.2831853 * gen_rand(payload.rand_state);
    const float u = 2.0 * gen_rand(payload.rand_state) - 1.0;
    const float r = sqrt(1.0 - u * u);

    payload.color = hit_result.color;
    payload.ray_origin = hit_result.position + 0.0001 * hit_result.normal;
    payload.ray_direction = normalize(hit_result.normal + vec3(r * cos(phi), r * sin(phi), u));
    payload.ray_hit_sky = false;
}
