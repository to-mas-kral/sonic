#ifndef PT_MATH_UTILS_H
#define PT_MATH_UTILS_H

#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "../utils/numtypes.h"

const f32 EPS = 0.00001f;

__device__ f32 safe_sqrt(f32 v) {
    // Sanity check
    assert(v >= -EPS);

    return sqrt(max(0.f, v));
}

template <typename T> __device__ f32 sqr(T v) { return v * v; }

#endif // PT_MATH_UTILS_H
