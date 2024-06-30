#ifndef PT_COMMON_H
#define PT_COMMON_H

#include "../color/spectrum.h"
#include "../integrator/shading_frame.h"
#include "../scene/texture.h"
#include "../utils/basic_types.h"

#include <complex>

/// Taken from PBRTv4
f32
fresnel_dielectric(f32 rel_ior, f32 cos_theta_i);

/// Adapted from PBRTv4
f32
fresnel_conductor(std::complex<f32> rel_ior, f32 cos_theta_i);

/// Adapted from PBRTv4
std::optional<vec3>
refract(const ShadingFrameIncomplete &sframe, const norm_vec3 &wo, f32 rel_ior);

inline f32
fetch_alpha(const FloatTexture *texture, const vec2 &uv) {
    return std::max(texture->fetch(uv), 0.01f);
}

inline spectral
fetch_reflectance(const SpectrumTexture *texture, const vec2 &uv,
                  const SampledLambdas &lambdas) {
    auto refl = texture->fetch(uv, lambdas);

    // This has to be done due to the PBRT format accepting textures with reflectance
    // potentially being > 1
    refl.clamp(0.f, 1.f);

    return refl;
}

#endif // PT_COMMON_H
