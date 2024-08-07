
#ifndef COORDINATE_SYSTEM_H
#define COORDINATE_SYSTEM_H

#include "vecmath.h"

#include <tuple>

inline std::tuple<vec3, vec3, vec3>
coordinate_system(norm_vec3 v1) {
    f32 sign = std::copysign(1.f, v1.z);
    f32 a = -1.f / (sign + v1.z);
    f32 b = v1.x * v1.y * a;

    vec3 v2 = vec3(1.f + sign * sqr(v1.x) * a, sign * b, -sign * v1.x);
    vec3 v3 = vec3(b, sign + sqr(v1.y) * a, -v1.y);

    assert(std::abs(vec3::dot(v1, v2)) < 0.00001f);
    assert(std::abs(vec3::dot(v1, v3)) < 0.00001f);
    assert(std::abs(vec3::dot(v2, v3)) < 0.00001f);

    return {v1, v2, v3};
}

class CoordinateSystem {
public:
    explicit
    CoordinateSystem(const norm_vec3 &p_z) {
        auto [b0, b1, b2] = coordinate_system(p_z);
        z = b0;
        x = b1;
        y = b2;
    }

    vec3
    to_local(const norm_vec3 &input) const {
        return vec3(vec3::dot(input, x), vec3::dot(input, y), vec3::dot(input, z));
    }

    vec3
    from_local(const norm_vec3 &input) const {
        return input.x * x + input.y * y + input.z * z;
    }

private:
    vec3 x{};
    vec3 y{};
    vec3 z{};
};

#endif // COORDINATE_SYSTEM_H
