#ifndef PT_COMMON_H
#define PT_COMMON_H

#include "../color/spectrum.h"
#include "../utils/basic_types.h"

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

#endif // PT_COMMON_H
