#include "common.h"

#include "../integrator/shading_frame.h"

#include <cmath>

f32
fresnel_dielectric(const f32 rel_ior, const f32 cos_theta_i) {
    const f32 sin2_theta_i = 1.F - sqr(cos_theta_i);
    const f32 sin2_theta_t = sin2_theta_i / sqr(rel_ior);
    if (sin2_theta_t >= 1.F) {
        // Total internal reflection
        return 1.F;
    } else {
        const f32 cos_theta_t = std::sqrt(1.F - sin2_theta_t);

        const f32 r_parl =
            (rel_ior * cos_theta_i - cos_theta_t) / (rel_ior * cos_theta_i + cos_theta_t);
        const f32 r_perp =
            (cos_theta_i - rel_ior * cos_theta_t) / (cos_theta_i + rel_ior * cos_theta_t);
        return (sqr(r_parl) + sqr(r_perp)) / 2.F;
    }
}

f32
fresnel_conductor(const std::complex<f32> rel_ior, const f32 cos_theta_i) {
    using complex = std::complex<f32>;
    const f32 sin2_theta_i = 1.F - sqr(cos_theta_i);
    const complex sin2_theta_t = sin2_theta_i / sqr(rel_ior);

    const complex cos_theta_t = std::sqrt(1.F - sin2_theta_t);
    const complex r_parl =
        (rel_ior * cos_theta_i - cos_theta_t) / (rel_ior * cos_theta_i + cos_theta_t);
    const complex r_perp =
        (cos_theta_i - rel_ior * cos_theta_t) / (cos_theta_i + rel_ior * cos_theta_t);
    return (norm(r_parl) + norm(r_perp)) / 2.F;
}

std::optional<vec3>
refract(const norm_vec3 &wo, const f32 rel_ior) {
    const f32 cos_theta_i = ShadingFrameIncomplete::cos_theta(wo);
    const f32 sin2_theta_i = std::max(0.F, 1.F - sqr(cos_theta_i));
    const f32 sin2_theta_t = sin2_theta_i / sqr(rel_ior);

    // Total internal reflection
    if (sin2_theta_t >= 1.F) {
        return {};
    }

    const f32 cos_theta_t = std::sqrt(1.F - sin2_theta_t);
    return (-wo / rel_ior) +
           vec3(0.F, 0.F, 1.F) * ((cos_theta_i / rel_ior) - cos_theta_t);
}
