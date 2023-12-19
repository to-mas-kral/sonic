
#include "framebuffer.h"

Framebuffer::Framebuffer(u32 image_x, u32 image_y) : image_x(image_x), image_y(image_y) {
    pixels = UmVector<vec3>(vec3(0.f, 0.f, 0.f), num_pixels());
}
