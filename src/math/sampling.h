#ifndef PT_SAMPLING_H
#define PT_SAMPLING_H

#include <cuda/std/cmath>
#include <curand_kernel.h>

#include "../math/math_utils.h"
#include "../utils/numtypes.h"
#include "../utils/rng.h"

/*
 * All sampling code was taken from Physically Based Rendering, 4th edition
 * */

__device__ vec2 sample_uniform_disk_concentric(vec2 u) {
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

__device__ vec3 sample_cosine_hemisphere(curandState *rand_state) {
    f32 u = rng(rand_state);
    f32 v = rng(rand_state);

    vec2 d = sample_uniform_disk_concentric(vec2(u, v));
    f32 z = safe_sqrt(1.f - d.x * d.x - d.y * d.y);
    return vec3(d.x, d.y, z);
}

#endif // PT_SAMPLING_H
