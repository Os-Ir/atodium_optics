#version 460
#extension GL_EXT_ray_tracing: require
#extension GL_GOOGLE_include_directive: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_scalar_block_layout: enable

#include "rt.h.glsl"

layout (location = 0) rayPayloadInEXT PayloadInfo payload;

vec3 sky_color(const vec3 direction) {
    if (direction.y > 0.0f) {
        return mix(vec3(1.0f), vec3(0.25f, 0.5f, 1.0f), direction.y);
    } else {
        return vec3(0.03f);
    }
}

void main() {
    vec3 ray_direction = gl_WorldRayDirectionEXT;

    payload.color = sky_color(ray_direction);
    payload.ray_hit_sky = true;
}
