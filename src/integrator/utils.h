#ifndef PT_UTILS_H
#define PT_UTILS_H

#include <cuda/std/optional>

#include "../utils/basic_types.h"

/// Randomly selects if a path should be terminated based on its throughput.
/// Roulette is only applied after the first 3 bounces.
/// Returns true if path should be terminated. If not, also returns roulette compensation.
__device__ __forceinline__ COption<f32>
russian_roulette(u32 depth, f32 u, const vec3 &throughput) {
    if (depth > 3) {
        f32 survival_prob =
            1.f - max(glm::max(throughput.x, throughput.y, throughput.z), 0.05f);

        if (u < survival_prob) {
            return {};
        } else {
            f32 roulette_compensation = 1.f - survival_prob;
            return roulette_compensation;
        }
    } else {
        return 1.f;
    }
}

struct Intersection {
    u32 material_id;
    u32 light_id;
    bool has_light;
    vec3 normal;
    vec3 pos;
    vec2 uv;
};

__device__ __forceinline__ Ray
spawn_ray(Intersection &its, const vec3 &dir) {
    // TODO: more robust floating-point error handling when spawning rays
    vec3 ray_orig = its.pos + (0.0001f * its.normal);
    return Ray(ray_orig, dir);
}

struct ShadingGeometry {
    f32 cos_theta;
    /// Dot product between normal and w_o.
    f32 nowo;
    /// Dot product between normal and halfway vector.
    f32 noh;
    /// Dot product between halfway vector and w_o.
    f32 howo;
    /// Halfway vector
    vec3 h;
};

///  Following PBRT, w_i is incident direction and w_o is outgoing direction.
///  w_o goes "towards the viewer" and w_i "towards the light"
__device__ __forceinline__ ShadingGeometry
get_shading_geom(const vec3 &normal, const vec3 &w_i, const vec3 &w_o) {
    // TODO: what to do when cos_theta is 0 ? this minimum value is a band-aid
    f32 cos_theta = max(glm::dot(normal, w_i), 0.0001f);
    vec3 h = glm::normalize(w_i + w_o);
    f32 noh = glm::dot(normal, h);
    f32 nowo = glm::dot(normal, w_o);
    f32 howo = glm::dot(h, w_o);

    return ShadingGeometry{
        .cos_theta = cos_theta,
        .nowo = nowo,
        .noh = noh,
        .howo = howo,
        .h = h,
    };
}

/// Specific case where 1 sample is taken from each distribution.
__device__ __forceinline__ f32
mis_power_heuristic(f32 fpdf, f32 gpdf) {
    return sqr(fpdf) / (sqr(fpdf) + sqr(gpdf));
}

#endif // PT_UTILS_H
