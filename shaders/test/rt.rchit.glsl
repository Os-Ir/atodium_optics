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

hitAttributeEXT vec2 hit_uv;

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

HitResult get_hit_result() {
    const int primitive_id = gl_PrimitiveID;

    const uint i0 = indices.data[3 * primitive_id + 0];
    const uint i1 = indices.data[3 * primitive_id + 1];
    const uint i2 = indices.data[3 * primitive_id + 2];

    const vec3 v0 = vertices.data[i0].xyz;
    const vec3 v1 = vertices.data[i1].xyz;
    const vec3 v2 = vertices.data[i2].xyz;

    const vec3 barycentrics = vec3(1.0 - hit_uv.x - hit_uv.y, hit_uv.xy);

    const vec3 object_normal = cross(v1 - v0, v2 - v0);

    vec3 normal = normalize((object_normal).xyz);

    HitResult result;
    result.position = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
    result.normal = faceforward(normal, gl_WorldRayDirectionEXT, normal);
    result.color = vec3(0.7f);

    return result;
}

void main() {
    HitResult hit_result = get_hit_result();

    const float phi = 6.2831853 * gen_rand(payload.rand_state);
    const float u = 2.0 * gen_rand(payload.rand_state) - 1.0;
    const float r = sqrt(1.0 - u * u);

    payload.color = hit_result.color;
    payload.ray_origin = hit_result.position + 0.0001 * hit_result.normal;
    payload.ray_direction = normalize(hit_result.normal + vec3(r * cos(phi), r * sin(phi), u));
    payload.ray_hit_sky = false;
}
