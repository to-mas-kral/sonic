#include "sampling.h"

#include "../utils/algs.h"

#include <spdlog/spdlog.h>

vec2
sample_uniform_disk_concentric(const vec2 &u) {
    const vec2 u_offset = (2.f * u) - vec2(1.f, 1.f);

    if (u_offset.x == 0.f && u_offset.y == 0.f) {
        return vec2(0.f, 0.f);
    }

    f32 theta;
    f32 r;
    if (std::abs(u_offset.x) > std::abs(u_offset.y)) {
        r = u_offset.x;
        theta = (M_PIf / 4.f) * (u_offset.y / u_offset.x);
    } else {
        r = u_offset.y;
        theta = (M_PIf / 2.f) - (M_PIf / 4.f) * (u_offset.x / u_offset.y);
    }
    return r * vec2(std::cos(theta), std::sin(theta));
}

norm_vec3
sample_cosine_hemisphere(const vec2 &sample) {
    const vec2 d = sample_uniform_disk_concentric(sample);
    const f32 z = safe_sqrt(1.f - d.x * d.x - d.y * d.y);
    return norm_vec3(d.x, d.y, z);
}

norm_vec3
sample_uniform_sphere(const vec2 &sample) {
    const f32 z = 1.f - 2.f * sample.x;
    const f32 r = std::sqrt(std::max(1.f - sqr(z), 0.f));
    const f32 phi = 2.f * M_PIf * sample.y;
    return vec3(r * std::cos(phi), r * std::sin(phi), z).normalized();
}

norm_vec3
sample_uniform_hemisphere(const vec2 &sample) {
    const f32 z = sample.x;
    const f32 r = std::sqrt(std::max(1.f - sqr(z), 0.f));
    const f32 phi = 2 * M_PIf * sample.y;
    return vec3(r * std::cos(phi), r * std::sin(phi), z).normalized();
}

vec3
sample_uniform_triangle(const vec2 &sample) {
    const f32 sqrt_u = std::sqrt(sample.x);

    const f32 b0 = 1.f - sqrt_u;
    const f32 b1 = sample.y * sqrt_u;
    const f32 b2 = 1.f - b0 - b1;

    assert(b0 + b1 + b2 == 1.f);

    return vec3(b0, b1, b2);
}
