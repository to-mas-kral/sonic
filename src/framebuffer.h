#ifndef PT_FRAMEBUFFER_H
#define PT_FRAMEBUFFER_H

#include <curand_kernel.h>

#include "utils/cuda_err.h"
#include "utils/numtypes.h"
#include "utils/shared_vector.h"

// TODO: might be better to just copy come PCG implementation, this uses a LOT of space
// and init time..
__global__ void init_rand_state(u32 pixels, curandState *curand_state) {
    u32 pixel = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pixel < pixels) {
        curand_init(1984 + pixel, 0, 0, &curand_state[pixel]);
    }
}

class Framebuffer {
public:
    Framebuffer() : image_x{0}, image_y{0} {};

    Framebuffer(u32 image_x, u32 image_y) : image_x(image_x), image_y(image_y) {
        pixels = SharedVector<vec3>(vec3(0.f, 0.f, 0.f), num_pixels());
        rand_state = SharedVector<curandState>(num_pixels());

        const u32 threads_per_block = 256;
        u32 blocks = (num_pixels() + threads_per_block - 1U) / threads_per_block;

        // RenderContext isn't initialized yet. Have to pass rand_state raw pointer,
        // because rand_state itself could still be placed on the stack at this point.
        init_rand_state<<<blocks, threads_per_block>>>(num_pixels(),
                                                       rand_state.get_ptr());
        cudaDeviceSynchronize();
        CUDA_CHECK_LAST_ERROR();

        rand_state.assume_all_init();
    }

    __host__ __device__ u64 pixel_index(u64 x, u64 y) const { return (y * image_x) + x; }
    __host__ __device__ u32 num_pixels() const { return image_x * image_y; }
    __host__ __device__ u32 get_image_x() const { return image_x; }
    __host__ __device__ u32 get_image_y() const { return image_y; }

    __host__ __device__ SharedVector<vec3> &get_pixels() { return pixels; }
    __host__ __device__ SharedVector<curandState> &get_rand_state() { return rand_state; }

private:
    SharedVector<vec3> pixels;
    SharedVector<curandState> rand_state;

    u32 image_x;
    u32 image_y;
};

#endif // PT_FRAMEBUFFER_H
