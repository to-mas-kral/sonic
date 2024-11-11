
#ifndef COORDINATE_SYSTEM_H
#define COORDINATE_SYSTEM_H

#include "vecmath.h"

#include <tuple>

inline std::tuple<vec3, vec3, vec3>
coordinate_system(norm_vec3 v1) {
    const f32 sign = std::copysign(1.F, v1.z);
    const f32 a = -1.F / (sign + v1.z);
    const f32 b = v1.x * v1.y * a;

    const auto v2 = vec3(1.F + (sign * sqr(v1.x) * a), sign * b, -sign * v1.x);
    const auto v3 = vec3(b, sign + (sqr(v1.y) * a), -v1.y);

    assert(std::abs(vec3::dot(v1, v2)) < 0.00001F);
    assert(std::abs(vec3::dot(v1, v3)) < 0.00001F);
    assert(std::abs(vec3::dot(v2, v3)) < 0.00001F);

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
    vec3 x;
    vec3 y;
    vec3 z;
};

#endif // COORDINATE_SYSTEM_H
