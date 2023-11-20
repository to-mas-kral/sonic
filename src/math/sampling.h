#ifndef PT_SAMPLING_H
#define PT_SAMPLING_H

#include <cuda/std/cmath>
#include <curand_kernel.h>

#include "../math/math_utils.h"
#include "../utils/numtypes.h"
#include "../utils/rng.h"

#include <cuda/std/span>

/*
 * All sampling code was taken from Physically Based Rendering, 4th edition
 * */

__device__ inline vec2 sample_uniform_disk_concentric(const vec2 &u) {
    vec2 u_offset = (2.f * u) - vec2(1.f, 1.f);

    if (u_offset.x == 0.f && u_offset.y == 0.f) {
        return vec2(0.f, 0.f);
    }

    f32 theta;
    f32 r;
    if (abs(u_offset.x) > abs(u_offset.y)) {
        r = u_offset.x;
        theta = (M_PIf / 4.f) * (u_offset.y / u_offset.x);
    } else {
        r = u_offset.y;
        theta = (M_PIf / 2.f) - (M_PIf / 4.f) * (u_offset.x / u_offset.y);
    }
    return r * vec2(cos(theta), sin(theta));
}

__device__ inline vec3 sample_cosine_hemisphere(const vec2 &sample) {
    vec2 d = sample_uniform_disk_concentric(sample);
    f32 z = safe_sqrt(1.f - d.x * d.x - d.y * d.y);
    return vec3(d.x, d.y, z);
}

__device__ __forceinline__ vec3 sample_uniform_sphere(const vec2 &sample) {
    f32 z = 1.f - 2.f * sample.x;
    f32 r = sqrt(max(1.f - sqr(z), 0.f));
    f32 phi = 2.f * M_PIf * sample.y;
    return glm::normalize(vec3(r * cos(phi), r * sin(phi), z));
}

constexpr f32 UNIFORM_SPHERE_SAMPLE_PDF = M_PIf / 4.f;

/// Taken from PBRT - UniformSampleTriangle.
/// Return barycentric coordinates that can be used to sample any triangle.
__device__ __forceinline__ vec3 sample_uniform_triangle(const vec2 &sample) {
    f32 sqrt_u = sqrt(sample.x);

    f32 b0 = 1.f - sqrt_u;
    f32 b1 = sample.y * sqrt_u;
    f32 b2 = 1.f - b0 - b1;

    assert(b0 + b1 + b2 == 1.f);

    return vec3(b0, b1, b2);
}

/// Samples a CMF, return an index into the CMF slice.
/// Expects a normalized CMF.
__device__ __forceinline__ u32 sample_discrete_cmf(const cuda::std::span<f32> cmf,
                                                   f32 sample) {
    // TODO: optimization - binary search
    for (u32 i = 0; i < cmf.size(); i++) {
        if (sample < cmf[i]) {
            return i;
        }
    }

    assert(false);
}

#endif // PT_SAMPLING_H
