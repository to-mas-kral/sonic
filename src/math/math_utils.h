#ifndef PT_MATH_UTILS_H
#define PT_MATH_UTILS_H

#include "../utils/basic_types.h"

#include <cassert>
#include <cmath>

static constexpr f32 EPS = 0.00001F;

const f32 ONE_MINUS_EPS = std::nexttoward(1.F, 0.F);

template <std::floating_point T>
T
safe_sqrt(T v) {
    assert(v >= static_cast<T>(-EPS));

    return std::sqrt(std::max(static_cast<T>(0.F), v));
}

template <class T>
concept Squarable = requires(T a) { a *a; };

template <Squarable T>
constexpr T
sqr(T v) {
    return v * v;
}

template <std::floating_point T>
T
to_rad(T v) {
    return v * M_PIf / 180.F;
}

template <std::floating_point T, class... Args>
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

template <class T>
concept Lerpable = requires(T a, T b) { (a * 0.5F) + (b * 0.5F); };

template <Lerpable T>
T
lerp(f32 t, const T &start, const T &end) {
    return (start * (1.F - t)) + (end * t);
}

#endif // PT_MATH_UTILS_H
