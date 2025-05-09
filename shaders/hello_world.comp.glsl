#version 460

layout (local_size_x = 16, local_size_y = 8, local_size_z = 1) in;

layout (set = 0, binding = 0) buffer storageBuffer {
    float imageData[];
} storageImage;

void main() {
    const uvec2 resolution = uvec2(800, 600);

    const uvec2 pixel = gl_GlobalInvocationID.xy;

    if ((pixel.x >= resolution.x) || (pixel.y >= resolution.y)) {
        return;
    }

    const vec3 pixelColor = vec3(float(pixel.x) / resolution.x, float(pixel.y) / resolution.y, 0.0);

    uint linearIndex = resolution.x * pixel.y + pixel.x;

    storageImage.imageData[3 * linearIndex + 0] = pixelColor.r;
    storageImage.imageData[3 * linearIndex + 1] = pixelColor.g;
    storageImage.imageData[3 * linearIndex + 2] = pixelColor.b;
}
