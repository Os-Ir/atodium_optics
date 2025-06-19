#version 460

layout (local_size_x = 16, local_size_y = 8, local_size_z = 1) in;

layout (set = 0, binding = 0, scalar) buffer StorageImage {
    vec4 data[];
} storage_image;

void main() {
    const uvec2 resolution = uvec2(800, 600);

    const uvec2 pixel = gl_GlobalInvocationID.xy;

    if ((pixel.x >= resolution.x) || (pixel.y >= resolution.y)) {
        return;
    }

    const vec3 pixel_color = vec3(float(pixel.x) / resolution.x, float(pixel.y) / resolution.y, 0.0);

    uint linear_idx = resolution.x * pixel.y + pixel.x;

    storage_image.data[linear_idx] = vec4(pixel_color, 1.0);
}
