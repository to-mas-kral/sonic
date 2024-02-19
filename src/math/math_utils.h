#ifndef PT_MATH_UTILS_H
#define PT_MATH_UTILS_H

#include "../utils/basic_types.h"

#include <cassert>
#include <cmath>

static constexpr f32 EPS = 0.00001f;

template <typename T>
T
safe_sqrt(T v) {
    assert(v >= static_cast<T>(-EPS));

    return std::sqrt(std::max(static_cast<T>(0.), v));
}

template <typename T>
T
sqr(T v) {
    return v * v;
}

template <typename T>
T
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
T
lerp(f32 t, const T &start, const T &end) {
    return start * (1.f - t) + end * t;
}

#endif // PT_MATH_UTILS_H
