#ifndef PT_AABB_H
#define PT_AABB_H

#include <cuda/std/limits>

#include "../utils/numtypes.h"
#include "axis.h"
#include "ray.h"

class AABB {
public:
    vec3 min;
    vec3 max;

    AABB() {
        min = vec3(cuda::std::numeric_limits<f32>::max());
        max = vec3(cuda::std::numeric_limits<f32>::lowest());
    }

    AABB(const vec3 &a, const vec3 &b) {
        min = glm::min(a, b);
        max = glm::max(a, b);
    }

    /// Taken from PBRTv4.
    /// Original: An Efficient and Robust Ray–Box Intersection Algorithm.
    /// https://people.csail.mit.edu/amy/papers/box-jgt.pdf
    __host__ __device__ bool intersects(const Ray &ray, f32 ray_tmax,
                                        const vec3 &inv_ray_dir,
                                        const bvec3 &dir_is_neg) const {
        f32 tmin = ((*this)[dir_is_neg.x].x - ray.o.x) * inv_ray_dir.x;
        f32 tmax = ((*this)[!dir_is_neg.x].x - ray.o.x) * inv_ray_dir.x;
        f32 tymin = ((*this)[dir_is_neg.y].y - ray.o.y) * inv_ray_dir.y;
        f32 tymax = ((*this)[!dir_is_neg.y].y - ray.o.y) * inv_ray_dir.y;

        if (tmin > tymax || tymin > tmax) {
            return false;
        }
        if (tymin > tmin) {
            tmin = tymin;
        }
        if (tymax < tmax) {
            tmax = tymax;
        }

        f32 tzmin = ((*this)[dir_is_neg.z].z - ray.o.z) * inv_ray_dir.z;
        f32 tzmax = ((*this)[!dir_is_neg.z].z - ray.o.z) * inv_ray_dir.z;

        if (tmin > tzmax || tzmin > tmax) {
            return false;
        }
        if (tzmin > tmin) {
            tmin = tzmin;
        }
        if (tzmax < tmax) {
            tmax = tzmax;
        }

        return (tmin < ray_tmax) && (tmax > 0.f);
    }

    AABB union_point(const vec3 &b) {
        AABB uni{};
        uni.min = glm::min(min, b);
        uni.max = glm::max(max, b);
        return uni;
    }

    AABB union_aabb(const AABB &b) {
        AABB uni{};
        uni.min = glm::min(min, b.min);
        uni.max = glm::max(max, b.max);
        return uni;
    }

    /// If *this* fits within *other*
    bool fits_within(const AABB &other) const {
        return glm::all(glm::greaterThanEqual(min, other.min)) &&
               glm::all(glm::lessThanEqual(max, other.max));
    }

    vec3 diagonal() const { return max - min; }

    /// Offset from 0..1 of a point inside this AABB
    vec3 offset_of(const vec3 &other) const {
        vec3 off = other - min;
        if (max.x > min.x) {
            off.x /= max.x - min.x;
        }
        if (max.y > min.y) {
            off.y /= max.y - min.y;
        }
        if (max.z > min.z) {
            off.z /= max.z - min.z;
        }
        return off;
    }

    f32 area() const {
        vec3 d = diagonal();
        return 2.f * (d.x * d.y + d.x * d.z + d.z * d.y);
    }

    vec3 center() const { return (min + max) / 2.f; }

    u8 max_axis() const {
        vec3 diag = diagonal();
        if (diag.x > diag.y && diag.x > diag.z) {
            return AXIS_X;
        } else if (diag.y > diag.z) {
            return AXIS_Y;
        } else {
            return AXIS_Z;
        }
    }

    bool is_empty() const { return min == max; }

    __host__ __device__ const vec3 operator[](bool idx) const {
        if (idx) {
            return max;
        } else {
            return min;
        }
    }
};
#endif // PT_AABB_H
