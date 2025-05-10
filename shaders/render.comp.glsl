#version 460

#extension GL_EXT_scalar_block_layout: require
#extension GL_EXT_ray_query: require

layout (local_size_x = 16, local_size_y = 8, local_size_z = 1) in;

layout (set = 0, binding = 0, scalar) buffer StorageImage {
    vec4 data[];
} storage_image;

layout (set = 0, binding = 1) uniform accelerationStructureEXT tlas;

layout (set = 0, binding = 2, scalar) buffer Vertices {
    vec4 data[];
} vertices;

layout (set = 0, binding = 3, scalar) buffer Indices {
    uint data[];
} indices;

struct HitResult {
    vec3 position;
    vec3 normal;
    vec3 color;
};

float gen_rand(inout uint rand_state) {
    rand_state = rand_state * 747796405 + 1;
    uint word = ((rand_state >> ((rand_state >> 28) + 4)) ^ rand_state) * 277803737;
    word = (word >> 22) ^ word;
    return float(word) / 4294967295.0f;
}

vec3 sky_color(vec3 direction) {
    if (direction.y > 0.0f) {
        return mix(vec3(1.0f), vec3(0.25f, 0.5f, 1.0f), direction.y);
    } else {
        return vec3(0.03f);
    }
}

HitResult get_hit_result(rayQueryEXT ray_query) {
    const int primitive_id = rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true);

    const uint i0 = indices.data[3 * primitive_id + 0];
    const uint i1 = indices.data[3 * primitive_id + 1];
    const uint i2 = indices.data[3 * primitive_id + 2];

    const vec3 v0 = vertices.data[i0].xyz;
    const vec3 v1 = vertices.data[i1].xyz;
    const vec3 v2 = vertices.data[i2].xyz;

    vec3 barycentrics = vec3(0.0, rayQueryGetIntersectionBarycentricsEXT(ray_query, true));
    barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;

    HitResult result;
    result.position = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
    result.normal = normalize(cross(v1 - v0, v2 - v0));
    result.color = vec3(0.7f);

    return result;
}

void main() {
    const uvec2 resolution = uvec2(800, 600);
    const uvec2 pixel = gl_GlobalInvocationID.xy;

    if ((pixel.x >= resolution.x) || (pixel.y >= resolution.y)) {
        return;
    }

    const vec3 camera_origin = vec3(-0.001, 1.0, 6.0);
    const float fov_vertical_slope = 1.0 / 5.0;
    const uint sample_level = 1024;
    const uint reflect_level = 32;
    const float t_min = 0.0;
    const float t_max = 10000.0;

    uint rand_state = resolution.x * pixel.y + pixel.x;
    vec3 integrated_color = vec3(0.0);

    for (uint sample_idx = 0; sample_idx < sample_level; sample_idx++) {
        const vec2 pixel_center = vec2(pixel) + vec2(gen_rand(rand_state), gen_rand(rand_state));
        const vec2 screen_uv = vec2((2.0 * pixel_center.x - resolution.x) / resolution.y, -(2.0 * pixel_center.y - resolution.y) / resolution.y);

        vec3 ray_origin = camera_origin;
        vec3 ray_direction = normalize(vec3(fov_vertical_slope * screen_uv.x, fov_vertical_slope * screen_uv.y, -1.0));

        vec3 current_ray_color = vec3(1.0);

        for (uint reflect_idx = 0; reflect_idx < reflect_level; reflect_idx++) {
            rayQueryEXT ray_query;
            rayQueryInitializeEXT(ray_query, tlas, gl_RayFlagsOpaqueEXT, 0xff, ray_origin, t_min, ray_direction, t_max);

            while (rayQueryProceedEXT(ray_query)) {}

            if (rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionTriangleEXT) {
                HitResult hit_result = get_hit_result(ray_query);
                hit_result.normal = faceforward(hit_result.normal, ray_direction, hit_result.normal);

                current_ray_color *= hit_result.color;

                const float phi = 6.2831853 * gen_rand(rand_state);
                const float u = 2.0 * gen_rand(rand_state) - 1.0;
                const float r = sqrt(1.0 - u * u);

                ray_origin = hit_result.position + 0.0001 * hit_result.normal;
                ray_direction = normalize(hit_result.normal + vec3(r * cos(phi), r * sin(phi), u));
            } else {
                current_ray_color *= sky_color(ray_direction);

                integrated_color += current_ray_color;
                break;
            }
        }
    }

    uint linear_idx = resolution.x * pixel.y + pixel.x;

    integrated_color = integrated_color / float(sample_level);

    storage_image.data[linear_idx] = vec4(integrated_color.r, integrated_color.g, integrated_color.b, 1.0);
}
