#ifndef PT_FRAMEBUFFER_H
#define PT_FRAMEBUFFER_H

#include <cuda/std/tuple>
#include <curand_kernel.h>

#include "utils/numtypes.h"
#include "utils/sampler.h"
#include "utils/shared_vector.h"

class Framebuffer {
public:
    Framebuffer() : image_x{0}, image_y{0} {};

    Framebuffer(u32 image_x, u32 image_y, dim3 blocks_dim, dim3 threads_dim);

    __device__ cuda::std::tuple<u64, u64> pixel_coords(dim3 block_dim, dim3 block_idx,
                                                       dim3 thread_idx) const;

    __device__ u64 pixel_index(u64 x, u64 y) const;

    __device__ u64 pixel_index(dim3 block_dim, dim3 block_idx, dim3 thread_idx) const;

    __host__ __device__ u32 num_pixels() const { return image_x * image_y; }
    __host__ __device__ u32 get_res_x() const { return image_x; }
    __host__ __device__ u32 get_res_y() const { return image_y; }

    __host__ __device__ SharedVector<vec3> &get_pixels() { return pixels; }
    __device__ SharedVector<Sampler> &get_rand_state() { return samplers; }

private:
    SharedVector<vec3> pixels;
    SharedVector<Sampler> samplers;

    u32 image_x;
    u32 image_y;
};

__device__ __forceinline__ u64 Framebuffer::pixel_index(u64 x, u64 y) const {
    return ((image_y - 1U - y) * image_x) + x;
}

__device__ __forceinline__ u64 Framebuffer::pixel_index(dim3 block_dim, dim3 block_idx,
                                                        dim3 thread_idx) const {
    auto [x, y] = pixel_coords(block_dim, block_idx, thread_idx);
    return pixel_index(x, y);
}

#endif // PT_FRAMEBUFFER_H
