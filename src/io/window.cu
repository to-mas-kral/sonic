
#include "window.h"

#include <spdlog/spdlog.h>

#include <stdexcept>

Window::Window(int resx, int resy) {
    framebuffer = std::vector<u32>(resx * resy, 0);
    auto a = framebuffer.size();
    window = mfb_open_ex("CUDA/OptiX Path Tracer", resx, resy, 0);
    if (!window) {
        spdlog::error("Could not initialize minifb window");
        throw std::runtime_error("");
    }
}

vec3 reinhard(const vec3 &v) { return v / (1.0f + v); }

void Window::update(Framebuffer &fb, int samples) {
    for (int pi = 0; pi < fb.num_pixels(); pi++) {
        vec3 rgb = fb.get_pixels()[pi];
        rgb /= static_cast<f32>(samples);
        rgb = glm::pow(rgb, vec3(1. / 2.2));

        vec3 tonemapped = reinhard(rgb);
        u8 r = static_cast<u8>(tonemapped.x * 255.f);
        u8 g = static_cast<u8>(tonemapped.y * 255.f);
        u8 b = static_cast<u8>(tonemapped.z * 255.f);

        framebuffer[pi] = MFB_ARGB(255, r, g, b);
    }

    int state = mfb_update_ex(window, framebuffer.data(), fb.get_res_x(), fb.get_res_y());

    if (state < 0) {
        window = nullptr;
        spdlog::critical("MiniFB window state is < 0, TODO");
    }
}

void Window::close() { mfb_close(window); }
