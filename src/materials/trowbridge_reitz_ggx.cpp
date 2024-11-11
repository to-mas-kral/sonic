#include "trowbridge_reitz_ggx.h"

#include "../integrator/shading_frame.h"

#include <algorithm>

bool
TrowbridgeReitzGGX::is_alpha_effectively_zero(const f32 alpha) {
    return alpha < 0.001F;
}

vec3
TrowbridgeReitzGGX::sample_vndf_hemisphere(const vec2 u, const vec3 wi) {
    // sample a spherical cap in (-wi.z, 1]
    const float phi = 2.F * M_PIf * u.x;
    const float z = std::fma((1.F - u.y), (1.F + wi.z), -wi.z);
    const float sin_theta = std::sqrt(std::clamp(1.F - (z * z), 0.F, 1.F));
    const float x = sin_theta * std::cos(phi);
    const float y = sin_theta * std::sin(phi);
    const auto c = vec3(x, y, z);
    // compute halfway direction;
    const vec3 h = c + wi;
    // return without normalization as this is done later (see line 25)
    return h;
}

f32
TrowbridgeReitzGGX::D(const f32 noh, const f32 alpha) {
    if (noh < 0.F) {
        return 0.F;
    }

    const f32 asq = sqr(alpha);
    const f32 denom = (sqr(noh) * (asq - 1.F)) + 1.F;

    return asq / (M_PIf * sqr(denom));
}

f32
TrowbridgeReitzGGX::G1(const f32 now, const f32 how, const f32 alpha) {
    if (how / now <= 0.F) {
        return 0.F;
    }

    const f32 asq = sqr(alpha);
    const f32 denom = now + std::sqrt(asq + ((1.F - asq) * sqr(now)));
    return (2.F * now) / denom;
}

f32
TrowbridgeReitzGGX::pdf(const ShadingFrame &sframe, const f32 alpha) {
    if (sframe.noh() < 0.F) {
        return 0.F;
    }

    const f32 g1 = G1(sframe.nowo(), sframe.howo(), alpha);
    const f32 d = D(sframe.noh(), alpha);
    return g1 * d / (4.F * std::abs(sframe.nowo()));
}

f32
TrowbridgeReitzGGX::vndf_ggx(const ShadingFrame &sframe, const f32 alpha) {
    if (sframe.noh() < 0.F) {
        return 0.F;
    }

    const f32 g1 = G1(sframe.nowo(), sframe.howo(), alpha);
    const f32 d = D(sframe.noh(), alpha);
    assert(g1 >= 0.F && g1 <= 1.F);
    assert(d >= 0.F);

    return (g1 / std::abs(sframe.nowo())) * d * std::max(sframe.howo(), 0.F);
}

f32
TrowbridgeReitzGGX::visibility_smith_height_correlated_ggx(const f32 nowo, const f32 nowi,
                                                           const f32 alpha) {
    const f32 asq = alpha * alpha;
    const f32 NoVsq = nowo * nowo;
    const f32 NoLsq = nowi * nowi;

    const f32 denoml = nowi * std::sqrt(asq + (NoVsq * (1.F - asq)));
    const f32 denomv = nowo * std::sqrt(asq + (NoLsq * (1.F - asq)));

    // TODO: protect against division by zero
    return 0.5F / (denoml + denomv);
}

norm_vec3
TrowbridgeReitzGGX::sample(const norm_vec3 &wo, const vec2 &xi, const f32 alpha) {
    // warp to the hemisphere configuration
    const norm_vec3 wi_std = vec3(wo.x * alpha, wo.y * alpha, wo.z).normalized();
    // sample the hemisphere
    const vec3 wm_std = sample_vndf_hemisphere(xi, wi_std);
    // warp back to the ellipsoid configuration
    const norm_vec3 wm = vec3(wm_std.x * alpha, wm_std.y * alpha, wm_std.z).normalized();

    const norm_vec3 wi = vec3::reflect(wo, wm).normalized();
    return wi;
}
