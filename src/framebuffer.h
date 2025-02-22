#ifndef PT_FRAMEBUFFER_H
#define PT_FRAMEBUFFER_H

#include "math/vecmath.h"
#include "utils/basic_types.h"

#include <vector>

template <typename T>
concept AOVVar =
    std::is_trivially_copyable_v<T> &&
    (std::is_same_v<T, u32> || std::is_same_v<T, f32> || std::is_same_v<T, tuple3>) &&
    requires(T a) { a += a; };

// clang-format off
using AOV = std::variant<
    std::vector<u32>,
    std::vector<f32>,
    std::vector<tuple3>
>;
// clang-format off

class Framebuffer {
public:
    Framebuffer() : m_width_x{0}, m_height_y{0} {}

    Framebuffer(const u32 image_x, const u32 image_y)
        : m_width_x{image_x}, m_height_y{image_y} {
        pixels = std::vector<vec3>(num_pixels(), vec3(0.F));
    }

    u32
    num_pixels() const {
        return m_width_x * m_height_y;
    }

    u32
    width_x() const {
        return m_width_x;
    }

    u32
    height_y() const {
        return m_height_y;
    }

    void
    reset() {
        std::ranges::fill(pixels, vec3(0.F));
        m_aovs.clear();
        num_samples = 0;
    }

    std::vector<vec3> &
    get_pixels() {
        return pixels;
    }

    void
    add_to_pixel(const uvec2 &pixel, const vec3 &val) {
        const auto pixel_index = (pixel.y * m_width_x) + pixel.x;
        pixels[pixel_index] += val;
    }

    template <typename AOVVar>
    void
    add_aov(const uvec2 &pixel, const std::string_view name, const AOVVar &val) {
        if (!m_aovs.contains(name)) {
            m_aovs.insert({name, std::vector<AOVVar>(num_pixels())});
        }

        auto &aov_var = m_aovs.at(name);
        auto &aov = std::get<std::vector<AOVVar>>(aov_var);

        const auto pixel_index = (pixel.y * m_width_x) + pixel.x;
        aov[pixel_index] += val;
    }

    std::unordered_map<std::string_view, AOV> &aovs() {
        return m_aovs;
    }

    u32 num_samples{0};

private:
    std::vector<vec3> pixels;
    std::unordered_map<std::string_view, AOV> m_aovs;
    
    u32 m_width_x;
    u32 m_height_y;
};

#endif // PT_FRAMEBUFFER_H
