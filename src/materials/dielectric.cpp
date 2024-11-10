#include "dielectric.h"

#include "../integrator/utils.h"
#include "common.h"

f32
DielectricMaterial::pdf() {
    return 0.f;
}

spectral
DielectricMaterial::eval() {
    // This should only be evaluated during sampling
    return spectral::ZERO();
}

BSDFSample
DielectricMaterial::sample(const ShadingFrameIncomplete &sframe, const norm_vec3 &wo,
                           const vec2 &sample, SampledLambdas &lambdas, const vec2 &uv,
                           const bool is_frontfacing) const {
    const auto int_ior_s = m_int_ior->fetch(uv, lambdas);
    if (!int_ior_s.is_constant()) {
        lambdas.terminate_secondary();
    }
    const f32 int_ior = int_ior_s[0];

    const f32 ext_ior = m_ext_ior.eval_single(lambdas[0]);
    f32 rel_ior = int_ior / ext_ior;
    if (!is_frontfacing) {
        rel_ior = 1.f / rel_ior;
    }

    const f32 cos_theta_i = ShadingFrameIncomplete::cos_theta(wo);
    const f32 fresnel_refl = fresnel_dielectric(rel_ior, cos_theta_i);

    auto reflect = [&] {
        const auto wi = ShadingFrameIncomplete::reflect(wo);
        const auto sframe_complete = ShadingFrame(sframe, wi, wo);

        return BSDFSample{
            .bsdf = spectral::make_constant(fresnel_refl) / sframe_complete.nowi(),
            .pdf = fresnel_refl,
            .did_refract = false,
            .sframe = sframe_complete,
        };
    };

    if (sample.x < fresnel_refl) {
        return reflect();
    } else {
        const auto refr = refract(wo, rel_ior);
        if (refr.has_value()) {
            const auto wi = refr.value().normalized();
            const auto sframe_complete = ShadingFrame(sframe, wi, wo, true);

            const f32 transmittance = m_transmittance.eval_single(lambdas[0]);

            return BSDFSample{
                .bsdf = spectral::make_constant(1.f - fresnel_refl) * transmittance /
                        sframe_complete.abs_nowi(),
                .pdf = 1.f - fresnel_refl,
                .did_refract = true,
                .sframe = sframe_complete,
            };
        } else {
            // Total internal reflection
            return reflect();
        }
    }
}
