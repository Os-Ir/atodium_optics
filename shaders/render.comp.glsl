#version 460

#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_ray_query : require

layout (local_size_x = 16, local_size_y = 8, local_size_z = 1) in;

layout (set = 0, binding = 0, scalar) buffer StorageImage {
    vec4 data[];
} storageImage;

layout (set = 0, binding = 1) uniform accelerationStructureEXT tlas;

void main() {
    const uvec2 resolution = uvec2(800, 600);

    const uvec2 pixel = gl_GlobalInvocationID.xy;

    if ((pixel.x >= resolution.x) || (pixel.y >= resolution.y)) {
        return;
    }

    const vec3 cameraOrigin = vec3(-0.001, 1.0, 6.0);

    vec3 rayOrigin = cameraOrigin;

    const vec2 screenUV = vec2(2.0 * (float(pixel.x) + 0.5 - 0.5 * resolution.x) / resolution.y, -(2.0 * (float(pixel.y) + 0.5 - 0.5 * resolution.y) / resolution.y));

    const float fovVerticalSlope = 1.0 / 5.0;
    vec3 rayDirection = normalize(vec3(fovVerticalSlope * screenUV.x, fovVerticalSlope * screenUV.y, -1.0));

    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(rayQuery, tlas, gl_RayFlagsNoOpaqueEXT, 0xff, rayOrigin, 0.0, rayDirection, 10000.0);

    float numIntersections = 0.0;
    while (rayQueryProceedEXT(rayQuery)) {
        numIntersections += 1.0;
    }

    vec4 pixelColor = vec4(0.0, 0.0, 0.2, 1.0);

    pixelColor = vec4(numIntersections / 10.0, numIntersections / 10.0, numIntersections / 10.0, 1.0);

    uint linearIndex = resolution.x * pixel.y + pixel.x;
    storageImage.data[linearIndex] = pixelColor;
}
