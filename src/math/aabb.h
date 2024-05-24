
#ifndef AABB_H
#define AABB_H
#include "vecmath.h"

struct AABB {
    AABB(const vec3 &low, const vec3 &high) : low(low), high(high) {}

    std::tuple<vec3, f32>
    bounding_sphere() const {
        const auto center = (high + low) / 2.f;
        const auto radius = (high - center).length();
        return {center, radius};
    }

    vec3 low{};
    vec3 high{};
};

#endif // AABB_H
