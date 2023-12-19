#ifndef PT_FRAMEBUFFER_H
#define PT_FRAMEBUFFER_H

#include <cuda/std/tuple>

#include "utils/basic_types.h"
#include "utils/um_vector.h"

class Framebuffer {
public:
    Framebuffer() : image_x{0}, image_y{0} {};

    Framebuffer(u32 image_x, u32 image_y);

    __host__ __device__ u32
    num_pixels() const {
        return image_x * image_y;
    }
    __host__ __device__ u32
    get_res_x() const {
        return image_x;
    }
    __host__ __device__ u32
    get_res_y() const {
        return image_y;
    }

    __host__ __device__ UmVector<vec3> &
    get_pixels() {
        return pixels;
    }

private:
    UmVector<vec3> pixels;

    u32 image_x;
    u32 image_y;
};

#endif // PT_FRAMEBUFFER_H
