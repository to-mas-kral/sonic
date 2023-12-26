#ifndef PT_UTILS_H
#define PT_UTILS_H

#include <cuda/std/optional>

#include "../math/vecmath.h"
#include "../utils/basic_types.h"

/// Randomly selects if a path should be terminated based on its throughput.
/// Roulette is only applied after the first 3 bounces.
/// Returns true if path should be terminated. If not, also returns roulette compensation.
__device__ __forceinline__ COption<f32>
russian_roulette(u32 depth, f32 u, const spectral &throughput) {
    if (depth > 3) {
        f32 survival_prob = 1.f - max(throughput.max_component(), 0.05f);

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
    /// "shading" normal affected by interpolation or normal maps
    norm_vec3 normal;
    /// "true" normal, always perpendicular to the geometry, used for self-intersection
    /// avoidance
    norm_vec3 geometric_normal;
    point3 pos;
    vec2 uv;
};

__device__ __forceinline__ constexpr float
origin() {
    return 1.0f / 32.0f;
}

__device__ __forceinline__ constexpr float
float_scale() {
    return 1.0f / 65536.0f;
}

__device__ __forceinline__ constexpr float
int_scale() {
    return 256.0f;
}

// Taken from GPU Gems - Chapter 6 - A Fast and Robust Method for Avoiding
// Self-Intersection - Carsten Wächter and Nikolaus Binder - NVIDIA
/// Normal points outward for rays exiting the surface, else is flipped.
__device__ __forceinline__ point3
offset_ray(const point3 &p, const norm_vec3 &n) {
    ivec3 of_i(int_scale() * n.x, int_scale() * n.y, int_scale() * n.z);

    point3 p_i(__int_as_float(__float_as_int(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
               __int_as_float(__float_as_int(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
               __int_as_float(__float_as_int(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

    return point3(fabsf(p.x) < origin() ? p.x + float_scale() * n.x : p_i.x,
                  fabsf(p.y) < origin() ? p.y + float_scale() * n.y : p_i.y,
                  fabsf(p.z) < origin() ? p.z + float_scale() * n.z : p_i.z);
}

__device__ __forceinline__ Ray
spawn_ray(Intersection &its, const norm_vec3 &dir) {
    point3 offset_orig = offset_ray(its.pos, its.geometric_normal);
    return Ray(offset_orig, dir);
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
    norm_vec3 h;
};

///  Following PBRT, w_i is incident direction and w_o is outgoing direction.
///  w_o goes "towards the viewer" and w_i "towards the light"
__device__ __forceinline__ ShadingGeometry
get_shading_geom(const norm_vec3 &normal, const norm_vec3 &w_i, const norm_vec3 &w_o) {
    // TODO: what to do when cos_theta is 0 ? this minimum value is a band-aid
    /*The f() function performs the required coordinate frame conversion and then queries
     * the BxDF. The rare case in which the wo direction lies exactly in the surface’s
     * tangent plane often leads to not-a-number (NaN) values in BxDF implementations that
     * further propagate and may eventually contaminate the rendered image. The BSDF
     * avoids this case by immediately returning a zero-valued SampledSpectrum. */
    f32 cos_theta = max(vec3::dot(normal, w_i), 0.0001f);
    norm_vec3 h = (w_i + w_o).normalized();
    f32 noh = vec3::dot(normal, h);
    f32 nowo = vec3::dot(normal, w_o);
    f32 howo = vec3::dot(h, w_o);

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
