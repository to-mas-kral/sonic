
#ifndef AABB_H
#define AABB_H

#include "../utils/panic.h"
#include "axis.h"
#include "vecmath.h"

#include <tuple>

class AABB {
public:
    AABB(const vec3 &low, const vec3 &high) : low(low), high(high) {}

    std::tuple<vec3, f32>
    bounding_sphere() const {
        const auto center = (high + low) / 2.F;
        const auto radius = (high - center).length();
        return {center, radius};
    }

    vec3
    diagonal() const {
        return high - low;
    }

    Axis
    longest_axis() const {
        const auto diag = diagonal();
        if (diag.x > diag.y && diag.x > diag.z) {
            return Axis::X;
        } else if (diag.y > diag.z) {
            return Axis::Y;
        } else {
            return Axis::Z;
        }
    }

    bool
    contains(const point3 &pos) const {
        // clang-format off
        return
            pos.x >= (low.x - EPS) && pos.x <= (high.x + EPS) &&
            pos.y >= (low.y - EPS) && pos.y <= (high.y + EPS) &&
            pos.z >= (low.z - EPS) && pos.z <= (high.z + EPS);
        // clang-format off
    }

    /// Returns on which side of the axis that splits the AABB in the middle the position lies
    /// true - right, false - left
    bool
    right_of_split_axis(const point3 &pos, const Axis axis) const {
        switch (axis) {
        case Axis::X: {
            const auto middle = low.x + 0.5F * (high.x - low.x);
            return pos.x >= middle;
        }
        case Axis::Y: {
            const auto middle = low.y + 0.5F * (high.y - low.y);
            return pos.y >= middle;
        }
        case Axis::Z: {
            const auto middle = low.z + 0.5F * (high.z - low.z);
            return pos.z >= middle;
        }
        default:
            panic();
        }
    }

    AABB right_half(const Axis axis) const {
        switch (axis) {
        case Axis::X: {
            const auto middle = low.x + 0.5F * (high.x - low.x);
            auto next_low = low;
            next_low.x = middle;
            return AABB(next_low, high);
        }
        case Axis::Y: {
            const auto middle = low.y + 0.5F * (high.y - low.y);
            auto next_low = low;
            next_low.y = middle;
            return AABB(next_low, high);
        }
        case Axis::Z: {
            const auto middle = low.z + 0.5F * (high.z - low.z);
            auto next_low = low;
            next_low.z = middle;
            return AABB(next_low, high);
        }
        default:
            panic();
        }
    }

    AABB left_half(const Axis axis) const {
        switch (axis) {
        case Axis::X: {
            const auto middle = low.x + 0.5F * (high.x - low.x);
            auto next_high = high;
            next_high.x = middle;
            return AABB(low, next_high);
        }
        case Axis::Y: {
            const auto middle = low.y + 0.5F * (high.y - low.y);
            auto next_high = high;
            next_high.y = middle;
            return AABB(low, next_high);
        }
        case Axis::Z: {
            const auto middle = low.z + 0.5F * (high.z - low.z);
            auto next_high = high;
            next_high.z = middle;
            return AABB(low, next_high);
        }
        default:
            panic();
        }
    }

private:
    vec3 low;
    vec3 high;
};

#endif // AABB_H
