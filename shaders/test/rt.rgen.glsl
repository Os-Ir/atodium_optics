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

layout (location = 0) rayPayloadEXT PayloadInfo payload;

layout (set = 0, binding = 0) uniform accelerationStructureEXT tlas;

layout (set = 0, binding = 1, rgba32f) uniform image2D image_output;

layout (set = 0, binding = 2, scalar) buffer Vertices {
    vec4 data[];
} vertices;

layout (set = 0, binding = 3, scalar) buffer Indices {
    uint data[];
} indices;

float gen_rand(inout uint rand_state) {
    rand_state = rand_state * 747796405 + 1;
    uint word = ((rand_state >> ((rand_state >> 28) + 4)) ^ rand_state) * 277803737;
    word = (word >> 22) ^ word;
    return float(word) / 4294967295.0f;
}

void main() {
    const ivec2 resolution = imageSize(image_output);
    const ivec2 pixel = ivec2(gl_LaunchIDEXT.xy);

    if ((pixel.x >= resolution.x) || (pixel.y >= resolution.y)) {
        return;
    }

    const vec3 camera_origin = vec3(-0.001, 1.0, 6.0);
    const float fov_vertical_slope = 1.0 / 5.0;
    const uint sample_level = 1024;
    const uint reflect_level = 32;
    const float t_min = 0.0;
    const float t_max = 10000.0;

    payload.rand_state = uint(resolution.x * pixel.y + pixel.x);
    vec3 integrated_color = vec3(0.0);

    for (uint sample_idx = 0; sample_idx < sample_level; sample_idx++) {
        const vec2 pixel_center = vec2(pixel) + vec2(gen_rand(payload.rand_state), gen_rand(payload.rand_state));
        const vec2 screen_uv = vec2((2.0 * pixel_center.x - resolution.x) / resolution.y, -(2.0 * pixel_center.y - resolution.y) / resolution.y);

        vec3 ray_origin = camera_origin;
        vec3 ray_direction = normalize(vec3(fov_vertical_slope * screen_uv.x, fov_vertical_slope * screen_uv.y, -1.0));

        vec3 current_ray_color = vec3(1.0);

        for (uint reflect_idx = 0; reflect_idx < reflect_level; reflect_idx++) {
            traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, ray_origin, t_min, ray_direction, t_max, 0);

            current_ray_color *= payload.color;

            if (payload.ray_hit_sky) {
                integrated_color += current_ray_color;
                break;
            } else {
                ray_origin = payload.ray_origin;
                ray_direction = payload.ray_direction;
            }
        }
    }

    integrated_color = integrated_color / float(sample_level);

    imageStore(image_output, pixel, vec4(integrated_color, 1.0));
}
