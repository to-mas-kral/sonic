#ifndef PT_VECMATH_H
#define PT_VECMATH_H

#include "../math/math_utils.h"

/// Taken from: Building an Orthonormal Basis, Revisited
/// Tom Duff, James Burgess, Per Christensen, Christophe Hery, Andrew Kensler, Max Liani,
/// and Ryusuke Villemin
__device__ inline cuda::std::tuple<vec3, vec3, vec3>
coordinate_system(vec3 v1) {
    f32 sign = copysign(1.f, v1.z);
    f32 a = -1.f / (sign + v1.z);
    f32 b = v1.x * v1.y * a;

    vec3 v2 = vec3(1. + sign * sqr(v1.x) * a, sign * b, -sign * v1.x);
    vec3 v3 = vec3(b, sign + sqr(v1.y) * a, -v1.y);

    return {v1, v2, v3};
}

/// Transforms dir into the basis of the normal
__device__ inline vec3
orient_dir(vec3 dir, vec3 normal) {
    auto [_, b1, b2] = coordinate_system(normal);
    vec3 sample_dir = b1 * dir.x + b2 * dir.y + normal * dir.z;

    sample_dir = glm::normalize(sample_dir);

    if (glm::dot(normal, sample_dir) < 0.f) {
        // FIXME: it's usually really close to 0, unsure what to do here...
        sample_dir = -sample_dir;
    }

    return sample_dir;
}

#endif // PT_VECMATH_H
