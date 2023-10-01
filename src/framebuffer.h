#ifndef PT_FRAMEBUFFER_H
#define PT_FRAMEBUFFER_H

#include <vector>

#include "utils/numtypes.h"
#include "utils/shared_vector.h"

class Framebuffer {
public:
    Framebuffer() : image_x{0}, image_y{0} {};

    Framebuffer(u32 image_x, u32 image_y) : image_x(image_x), image_y(image_y) {
        pixels = std::vector<vec3>(num_pixels(), vec3(0.f, 0.f, 0.f));
    }

    u64 pixel_index(u64 x, u64 y) const { return (y * image_x) + x; }
    u32 num_pixels() const { return image_x * image_y; }
    u32 get_image_x() const { return image_x; }
    u32 get_image_y() const { return image_y; }

    std::vector<vec3> &get_pixels() { return pixels; }

private:
    std::vector<vec3> pixels;

    u32 image_x;
    u32 image_y;
};

#endif // PT_FRAMEBUFFER_H
