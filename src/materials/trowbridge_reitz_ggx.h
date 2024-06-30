#ifndef PT_TROWBRIDGE_REITZ_GGX_H
#define PT_TROWBRIDGE_REITZ_GGX_H

#include "../integrator/utils.h"
#include "bsdf_sample.h"

namespace TrowbridgeReitzGGX {

bool
is_alpha_effectively_zero(f32 alpha);

// Taken from "Sampling Visible GGX Normals with Spherical Caps - Jonathan Dupuy"
vec3
sample_vndf_hemisphere(vec2 u, vec3 wi);

// Adapted from "Sampling Visible GGX Normals with Spherical Caps - Jonathan Dupuy"
norm_vec3
sample(const norm_vec3 &wo, const vec2 &xi, f32 alpha);

f32
D(f32 noh, f32 alpha);

f32
G1(f32 now, f32 how, f32 alpha);

f32
pdf(const ShadingFrame &sframe, f32 alpha);

f32
vndf_ggx(const ShadingFrame &sframe, f32 alpha);

f32
visibility_smith_height_correlated_ggx(f32 nowo, f32 nowi, f32 alpha);

} // namespace TrowbridgeReitzGGX

#endif // PT_TROWBRIDGE_REITZ_GGX_H
