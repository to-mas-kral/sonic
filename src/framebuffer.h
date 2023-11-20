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

#endif // PT_FRAMEBUFFER_H
