#version 460

#extension GL_EXT_scalar_block_layout: require
#extension GL_EXT_nonuniform_qualifier: enable

struct Vertex {
    vec4 pos;
    vec3 normal;
    vec2 uv;
    vec4 color;
    vec4 tangent;
};

struct Material {
    uint diffuse_map;
    uint normal_map;
    uint metallic_roughness_map;
    uint occlusion_map;

    vec4 base_color;
    float metallic_factor;
    float roughness_factor;

    uint material_type;
    float material_property;
};

struct Mesh {
    uint vertex_buffer;
    uint index_buffer;
    uint material;
};

struct Light {
    vec4 color;
    vec3 pos;
    vec3 direction;
    vec3 intensity;
    float spot;
    vec3 att;
    uint type;
    vec3 intensity;
};


layout (set = 0, binding = 0) uniform sampler2D sampler_color[];

layout (set = 0, binding = 1) readonly buffer Vertices {
    Vertex data[];
} vertices[];

layout (set = 0, binding = 2) readonly buffer Indices {
    ivec3 data[];
} indices[];

layout (set = 0, binding = 3) readonly buffer Materials {
    Material data[];
} materials;

layout (set = 0, binding = 4) readonly buffer Meshes {
    Mesh data[];
} meshes;

layout (set = 0, binding = 5) readonly buffer Lights {
    Light data[];
} lights;
