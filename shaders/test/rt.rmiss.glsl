#version 460
#extension GL_EXT_ray_tracing: require
#extension GL_GOOGLE_include_directive: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_scalar_block_layout: enable

struct PayloadInfo {
    uint rand_state;
    vec3 color;
    vec3 ray_origin;
    vec3 ray_direction;
    bool ray_hit_sky;
};

layout (location = 0) rayPayloadInEXT PayloadInfo payload;

layout (set = 0, binding = 0) uniform accelerationStructureEXT tlas;

layout (set = 0, binding = 1, rgba32f) uniform image2D image_output;

layout (set = 0, binding = 2, scalar) buffer Vertices {
    vec4 data[];
} vertices;

layout (set = 0, binding = 3, scalar) buffer Indices {
    uint data[];
} indices;

vec3 sky_color(vec3 direction) {
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
