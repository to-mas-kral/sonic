#ifndef PT_COMMON_H
#define PT_COMMON_H

#include "../color/spectrum.h"
#include "../utils/basic_types.h"
#include "../scene/texture.h"
#include "../scene/texture_id.h"

#include <complex>

/// Taken from PBRTv4
f32
fresnel_dielectric(f32 rel_ior, f32 cos_theta_i);

/// Adapted from PBRTv4
f32
fresnel_conductor(std::complex<f32> rel_ior, f32 cos_theta_i);

/// Adapted from PBRTv4
Option<vec3>
refract(const norm_vec3 &wo, const norm_vec3 &normal, f32 rel_ior);

inline f32
fetch_alpha(const Texture *textures, TextureId tex_id, const vec2 &uv) {
    return std::max(textures[tex_id.inner].fetch_float(uv), 0.01f);
}

#endif // PT_COMMON_H
