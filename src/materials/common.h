#ifndef PT_COMMON_H
#define PT_COMMON_H

#include "../color/spectrum.h"
#include "../utils/basic_types.h"

/// Taken from PBRTv4
__host__ __device__ static f32
fresnel_dielectric(f32 rel_ior, f32 cos_theta_i) {
    f32 sin2_theta_i = 1.f - sqr(cos_theta_i);
    f32 sin2_theta_t = sin2_theta_i / sqr(rel_ior);
    if (sin2_theta_t >= 1.f) {
        // Total internal reflection
        return 1.f;
    } else {
        f32 cos_theta_t = sqrt(1.f - sin2_theta_t);

        f32 r_parl =
            (rel_ior * cos_theta_i - cos_theta_t) / (rel_ior * cos_theta_i + cos_theta_t);
        f32 r_perp =
            (cos_theta_i - rel_ior * cos_theta_t) / (cos_theta_i + rel_ior * cos_theta_t);
        return (sqr(r_parl) + sqr(r_perp)) / 2.f;
    }
}

#endif // PT_COMMON_H
