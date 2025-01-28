#ifndef PT_FRAMEBUFFER_H
#define PT_FRAMEBUFFER_H

#include "math/vecmath.h"
#include "utils/basic_types.h"

#include <vector>

class Framebuffer {
public:
    Framebuffer() : image_x{0}, image_y{0} {}

    Framebuffer(const u32 image_x, const u32 image_y)
        : image_x{image_x}, image_y{image_y} {
        pixels = std::vector<vec3>(num_pixels(), vec3(0.F));
    }

    u32
    num_pixels() const {
        return image_x * image_y;
    }

    u32
    get_res_x() const {
        return image_x;
    }

    u32
    get_res_y() const {
        return image_y;
    }

    void
    reset() {
        std::fill(pixels.begin(), pixels.end(), vec3(0.f));
        num_samples = 0;
    }

    std::vector<vec3> &
    get_pixels() {
        return pixels;
    }

    u32 num_samples{0};

private:
    std::vector<vec3> pixels;

    u32 image_x;
    u32 image_y;
};

#endif // PT_FRAMEBUFFER_H
