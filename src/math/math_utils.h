#ifndef PT_MATH_UTILS_H
#define PT_MATH_UTILS_H

#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "../utils/basic_types.h"

const f32 EPS = 0.00001f;

__device__ inline f32
safe_sqrt(f32 v) {
    // Sanity check
    assert(v >= -EPS);

    return sqrt(max(0.f, v));
}

template <typename T>
__host__ __device__ f32
sqr(T v) {
    return v * v;
}

template <typename T>
T __host__ __device__
to_rad(T v) {
    return v * M_PIf / 180.f;
}

template <typename T, class... Args>
T
avg(Args... args) {
    constexpr int num_args = sizeof...(args);
    static_assert(num_args > 0);

    T total{};
    for (auto value : {args...}) {
        total += value;
    }

    return total / static_cast<T>(num_args);
}

template <typename T>
__device__ __forceinline__ T
barycentric_interp(const vec3 &bar, const T &x, const T &y, const T &z) {
    return bar.x * x + bar.y * y + bar.z * z;
}

#endif // PT_MATH_UTILS_H
