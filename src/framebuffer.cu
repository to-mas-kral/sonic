
#include "framebuffer.h"

__global__ void
init_rand_state(u32 pixels, u32 image_x, Sampler *samplers) {
    // This is the same as pixel_index() which can't be used because Framebuffer object
    // is still being created.
    auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
    auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
    auto pixel_index = (y * image_x) + x;

    if (pixel_index < pixels) {
        curand_init(1984 + pixel_index, 0, 0, samplers[pixel_index].get_rand_state());
    }
}

Framebuffer::Framebuffer(u32 image_x, u32 image_y, dim3 blocks_dim, dim3 threads_dim)
    : image_x(image_x), image_y(image_y) {
    pixels = SharedVector<vec3>(vec3(0.f, 0.f, 0.f), num_pixels());
    samplers = SharedVector<Sampler>(num_pixels());

    // RenderContext isn't initialized yet. Have to pass rand_state raw pointer,
    // because rand_state itself could still be placed on the stack at this point.
    init_rand_state<<<blocks_dim, threads_dim>>>(num_pixels(), image_x,
                                                 samplers.get_ptr());
    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR();

    samplers.assume_all_init();
}
