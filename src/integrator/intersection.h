#ifndef PT_INTERSECTION_H
#define PT_INTERSECTION_H

#include "../geometry/ray.h"
#include "../materials/material_id.h"
#include "../math/vecmath.h"
#include "../utils/basic_types.h"

struct Intersection {
    static Intersection
    make_empty() {
        return Intersection{.material_id = MaterialId{0},
                            .light_id = 0,
                            .has_light = false,
                            .normal{0.f, 1.f, 0.f},
                            .geometric_normal{0.f, 1.f, 0.f},
                            .pos{0.f, 0.f, 0.f},
                            .uv{0.f, 0.f}};
    }

    MaterialId material_id;
    u32 light_id;
    bool has_light;
    /// "shading" normal affected by interpolation or normal maps
    norm_vec3 normal;
    /// "true" normal, always perpendicular to the geometry, used for self-intersection
    /// avoidance
    norm_vec3 geometric_normal;
    point3 pos;
    vec2 uv;
};

constexpr float
origin() {
    return 1.0f / 32.0f;
}

constexpr float
float_scale() {
    return 1.0f / 65536.0f;
}

constexpr float
int_scale() {
    return 256.0f;
}

// Taken from GPU Gems - Chapter 6 - A Fast and Robust Method for Avoiding
// Self-Intersection - Carsten Wächter and Nikolaus Binder - NVIDIA
/// Normal points outward for rays exiting the surface, else is flipped.
inline point3
offset_ray(const point3 &p, const norm_vec3 &n) {
    ivec3 of_i(int_scale() * n.x, int_scale() * n.y, int_scale() * n.z);

    point3 p_i(
        std::bit_cast<f32>(std::bit_cast<i32>(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
        std::bit_cast<f32>(std::bit_cast<i32>(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
        std::bit_cast<f32>(std::bit_cast<i32>(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

    return point3(fabsf(p.x) < origin() ? p.x + float_scale() * n.x : p_i.x,
                  fabsf(p.y) < origin() ? p.y + float_scale() * n.y : p_i.y,
                  fabsf(p.z) < origin() ? p.z + float_scale() * n.z : p_i.z);
}

inline Ray
spawn_ray(const point3 &pos, const norm_vec3 &spawn_ray_normal, const norm_vec3 &wi) {
    point3 offset_orig = offset_ray(pos, spawn_ray_normal);
    return Ray(offset_orig, wi);
}

#endif // PT_INTERSECTION_H
