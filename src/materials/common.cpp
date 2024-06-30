#include "common.h"

#include "../integrator/shading_frame.h"

#include <cmath>

f32
fresnel_dielectric(f32 rel_ior, f32 cos_theta_i) {
    f32 sin2_theta_i = 1.f - sqr(cos_theta_i);
    f32 sin2_theta_t = sin2_theta_i / sqr(rel_ior);
    if (sin2_theta_t >= 1.f) {
        // Total internal reflection
        return 1.f;
    } else {
        f32 cos_theta_t = std::sqrt(1.f - sin2_theta_t);

        f32 r_parl =
            (rel_ior * cos_theta_i - cos_theta_t) / (rel_ior * cos_theta_i + cos_theta_t);
        f32 r_perp =
            (cos_theta_i - rel_ior * cos_theta_t) / (cos_theta_i + rel_ior * cos_theta_t);
        return (sqr(r_parl) + sqr(r_perp)) / 2.f;
    }
}

f32
fresnel_conductor(std::complex<f32> rel_ior, f32 cos_theta_i) {
    using complex = std::complex<f32>;
    f32 sin2_theta_i = 1.f - sqr(cos_theta_i);
    complex sin2_theta_t = sin2_theta_i / sqr(rel_ior);

    complex cos_theta_t = std::sqrt(1.f - sin2_theta_t);
    complex r_parl =
        (rel_ior * cos_theta_i - cos_theta_t) / (rel_ior * cos_theta_i + cos_theta_t);
    complex r_perp =
        (cos_theta_i - rel_ior * cos_theta_t) / (cos_theta_i + rel_ior * cos_theta_t);
    return (norm(r_parl) + norm(r_perp)) / 2.f;
}

std::optional<vec3>
refract(const ShadingFrameIncomplete &sframe, const norm_vec3 &wo, f32 rel_ior) {
    const f32 cos_theta_i = ShadingFrameIncomplete::cos_theta(wo);
    const f32 sin2_theta_i = std::max(0.f, 1.f - sqr(cos_theta_i));
    const f32 sin2_theta_t = sin2_theta_i / sqr(rel_ior);

    // Total internal reflection
    if (sin2_theta_t >= 1.f) {
        return {};
    }

    const f32 cos_theta_t = std::sqrt(1.f - sin2_theta_t);
    return (-wo / rel_ior) +
           vec3(0.f, 0.f, 1.f) * ((cos_theta_i / rel_ior) - cos_theta_t);
}
