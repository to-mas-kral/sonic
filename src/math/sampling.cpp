#include "sampling.h"

#include "../utils/algs.h"

#include <spdlog/spdlog.h>

vec2
sample_uniform_disk_concentric(const vec2 &u) {
    const vec2 u_offset = (2.F * u) - vec2(1.F, 1.F);

    if (u_offset.x == 0.F && u_offset.y == 0.F) {
        return vec2(0.F, 0.F);
    }

    f32 theta = 0.F;
    f32 r = 0.F;
    if (std::abs(u_offset.x) > std::abs(u_offset.y)) {
        r = u_offset.x;
        theta = (M_PIf / 4.F) * (u_offset.y / u_offset.x);
    } else {
        r = u_offset.y;
        theta = (M_PIf / 2.F) - (M_PIf / 4.F) * (u_offset.x / u_offset.y);
    }
    return r * vec2(std::cos(theta), std::sin(theta));
}

norm_vec3
sample_cosine_hemisphere(const vec2 &sample) {
    const vec2 d = sample_uniform_disk_concentric(sample);
    const f32 z = safe_sqrt(1.F - (d.x * d.x) - (d.y * d.y));
    return norm_vec3(d.x, d.y, z);
}

norm_vec3
sample_uniform_sphere(const vec2 &sample) {
    const f32 z = 1.F - (2.F * sample.x);
    const f32 r = std::sqrt(std::max(1.F - sqr(z), 0.F));
    const f32 phi = 2.F * M_PIf * sample.y;
    return vec3(r * std::cos(phi), r * std::sin(phi), z).normalized();
}

norm_vec3
sample_uniform_hemisphere(const vec2 &sample) {
    const f32 z = sample.x;
    const f32 r = std::sqrt(std::max(1.F - sqr(z), 0.F));
    const f32 phi = 2 * M_PIf * sample.y;
    return vec3(r * std::cos(phi), r * std::sin(phi), z).normalized();
}

vec3
sample_uniform_triangle(const vec2 &sample) {
    const f32 sqrt_u = std::sqrt(sample.x);

    const f32 b0 = 1.F - sqrt_u;
    const f32 b1 = sample.y * sqrt_u;
    const f32 b2 = 1.F - b0 - b1;

    assert(b0 + b1 + b2 == 1.F);

    return vec3(b0, b1, b2);
}
