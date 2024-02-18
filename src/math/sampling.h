#ifndef PT_SAMPLING_H
#define PT_SAMPLING_H

#include "../math/math_utils.h"
#include "../math/vecmath.h"
#include "../utils/basic_types.h"

/*
 * All sampling code was taken from Physically Based Rendering, 4th edition
 * */

inline vec2
sample_uniform_disk_concentric(const vec2 &u) {
    vec2 u_offset = (2.f * u) - vec2(1.f, 1.f);

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

inline norm_vec3
sample_cosine_hemisphere(const vec2 &sample) {
    vec2 d = sample_uniform_disk_concentric(sample);
    f32 z = safe_sqrt(1.f - d.x * d.x - d.y * d.y);
    return norm_vec3(d.x, d.y, z);
}

// z-up
inline vec3
sample_uniform_sphere(const vec2 &sample) {
    f32 z = 1.f - 2.f * sample.x;
    f32 r = std::sqrt(std::max(1.f - sqr(z), 0.f));
    f32 phi = 2.f * M_PIf * sample.y;
    return vec3(r * std::cos(phi), r * std::sin(phi), z).normalized();
}

// z-up
inline vec3
sample_uniform_hemisphere(const vec2 &sample) {
    f32 z = sample.x;
    f32 r = std::sqrt(std::max(1.f - sqr(z), 0.f));
    f32 phi = 2 * M_PIf * sample.y;
    return vec3(r * std::cos(phi), r * std::sin(phi), z).normalized();
}

/// Taken from PBRT - UniformSampleTriangle.
/// Return barycentric coordinates that can be used to sample any triangle.
inline vec3
sample_uniform_triangle(const vec2 &sample) {
    f32 sqrt_u = std::sqrt(sample.x);

    f32 b0 = 1.f - sqrt_u;
    f32 b1 = sample.y * sqrt_u;
    f32 b2 = 1.f - b0 - b1;

    assert(b0 + b1 + b2 == 1.f);

    return vec3(b0, b1, b2);
}

/// Samples a CMF, return an index into the CMF slice.
/// Expects a normalized CMF.
inline u32
sample_discrete_cmf(Span<f32> cmf, f32 sample) {
    // TODO: binary search
    for (u32 i = 0; i < cmf.size(); i++) {
        if (sample < cmf[i]) {
            return i;
        }
    }

    assert(false);
}

/// Samples a CMF, return a value in [0, 1), and an index into the CDF slice.
inline Tuple<f32, u32>
sample_continuous_cmf(Span<f32> cdf, f32 sample) {
    // TODO: binary search
    u32 offset = 0;
    for (u32 i = 0; i < cdf.size(); i++) {
        if (sample < cdf[i]) {
            offset = i;
            break;
        }
    }

    f32 du = sample - cdf[offset];
    if ((cdf[offset + 1] - cdf[offset]) > 0) {
        du /= (cdf[offset + 1] - cdf[offset]);
    }

    f32 res = (offset + du) / cdf.size();

    return {res, offset};
}

#endif // PT_SAMPLING_H
