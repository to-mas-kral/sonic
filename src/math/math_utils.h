#ifndef PT_MATH_UTILS_H
#define PT_MATH_UTILS_H

#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "../utils/numtypes.h"

const f32 EPS = 0.00001f;

__device__ inline f32 safe_sqrt(f32 v) {
    // Sanity check
    assert(v >= -EPS);

    return sqrt(max(0.f, v));
}

template <typename T> __device__ f32 sqr(T v) { return v * v; }

template <typename T> T to_rad(T v) { return v * M_PIf / 180.f; }

#endif // PT_MATH_UTILS_H
