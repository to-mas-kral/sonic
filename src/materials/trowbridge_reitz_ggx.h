#ifndef PT_TROWBRIDGE_REITZ_GGX_H
#define PT_TROWBRIDGE_REITZ_GGX_H

#include "bsdf_sample.h"

namespace TrowbridgeReitzGGX {
static bool
is_alpha_effectively_zero(f32 alpha) {
    return alpha < 0.001f;
}

// Taken from "Sampling Visible GGX Normals with Spherical Caps - Jonathan Dupuy"
__host__ __device__ inline vec3
sample_vndf_hemisphere(vec2 u, vec3 wi) {
    // sample a spherical cap in (-wi.z, 1]
    float phi = 2.f * M_PIf * u.x;
    float z = fma((1.f - u.y), (1.f + wi.z), -wi.z);
    float sin_theta = sqrt(cuda::std::clamp(1.f - z * z, 0.f, 1.f));
    float x = sin_theta * cos(phi);
    float y = sin_theta * sin(phi);
    vec3 c = vec3(x, y, z);
    // compute halfway direction;
    vec3 h = c + wi;
    // return without normalization as this is done later (see line 25)
    return h;
}

// Adapted from "Sampling Visible GGX Normals with Spherical Caps - Jonathan Dupuy"
__host__ __device__ inline norm_vec3
sample(const norm_vec3 &normal, const norm_vec3 &wo, const vec2 &ξ, f32 alpha) {
    auto [b0_, b1_, b2_] = coordinate_system(normal);
    auto bz = b0_.normalized();
    auto bx = b1_.normalized();
    auto by = b2_.normalized();

    norm_vec3 wo_sp =
        vec3(vec3::dot(wo, bx), vec3::dot(wo, by), vec3::dot(wo, bz)).normalized();

    // warp to the hemisphere configuration
    vec3 wi_std = vec3(wo_sp.x * alpha, wo_sp.y * alpha, wo_sp.z).normalized();
    // sample the hemisphere
    vec3 wm_std = sample_vndf_hemisphere(ξ, wi_std);
    // warp back to the ellipsoid configuration
    norm_vec3 wm = vec3(wm_std.x * alpha, wm_std.y * alpha, wm_std.z).normalized();

    norm_vec3 wm_rs = vec3(wm.x * bx + wm.y * by + wm.z * bz).normalized();
    norm_vec3 wi = vec3::reflect(wo, wm_rs).normalized();
    return wi;
}

static __host__ __device__ inline f32
D(f32 noh, f32 alpha) {
    if (noh < 0.f) {
        return 0.f;
    }

    f32 asq = sqr(alpha);
    f32 denom = sqr(noh) * (asq - 1.f) + 1.f;

    return asq / (M_PIf * sqr(denom));
}

static __host__ __device__ inline f32
G1(f32 now, f32 how, f32 alpha) {
    if (how / now <= 0.f) {
        return 0.f;
    }

    f32 asq = sqr(alpha);
    f32 denom = now + sqrt(asq + (1.f - asq) * sqr(now));
    return (2.f * now) / denom;
}

__host__ __device__ inline f32
pdf(const ShadingGeometry &sgeom, f32 alpha) {
    if (sgeom.noh < 0.f) {
        return 0.f;
    }

    f32 g1 = G1(sgeom.nowo, sgeom.howo, alpha);
    f32 d = D(sgeom.noh, alpha);
    return g1 * d / (4.f * abs(sgeom.nowo));
}

static __host__ __device__ inline f32
vndf_ggx(const ShadingGeometry &sgeom, f32 alpha) {
    if (sgeom.noh < 0.f) {
        return 0.f;
    }

    f32 g1 = G1(sgeom.nowo, sgeom.howo, alpha);
    f32 d = D(sgeom.noh, alpha);
    assert(g1 >= 0.f && g1 <= 1.f);
    assert(d >= 0.f);

    return (g1 / abs(sgeom.nowo)) * d * max(sgeom.howo, 0.f);
}

static __device__ inline f32
visibility_smith_height_correlated_ggx(f32 nowo, f32 nowi, f32 alpha) {
    f32 asq = alpha * alpha;
    f32 NoVsq = nowo * nowo;
    f32 NoLsq = nowi * nowi;

    f32 denoml = nowi * sqrt(asq + NoVsq * (1.f - asq));
    f32 denomv = nowo * sqrt(asq + NoLsq * (1.f - asq));

    // TODO: protect against division by zero
    return 0.5f / (denoml + denomv);
}

} // namespace TrowbridgeReitzGGX

#endif // PT_TROWBRIDGE_REITZ_GGX_H
