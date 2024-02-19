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
DielectricMaterial::sample(const norm_vec3 &normal, const norm_vec3 &wo,
                           const vec2 &sample, const SampledLambdas &lambdas,
                           const Texture *textures, const vec2 &uv,
                           bool is_frontfacing) const {
    f32 int_ior = m_int_ior.eval_single(lambdas[0]);
    f32 ext_ior = m_ext_ior.eval_single(lambdas[0]);
    f32 rel_ior = int_ior / ext_ior;
    if (!is_frontfacing) {
        rel_ior = 1.f / rel_ior;
    }

    f32 cos_theta_i = vec3::dot(wo, normal);
    f32 fresnel_refl = fresnel_dielectric(rel_ior, cos_theta_i);

    auto reflect = [&]() {
        auto wi = vec3::reflect(wo, normal).normalized();
        auto sgeom = ShadingGeometry::make(normal, wi, wo);

        return BSDFSample{
            .bsdf = spectral::make_constant(fresnel_refl) / sgeom.cos_theta,
            .wi = wi,
            .pdf = fresnel_refl,
            .did_refract = false,
        };
    };

    if (sample.x < fresnel_refl) {
        return reflect();
    } else {
        auto refr = refract(wo, normal, rel_ior);
        if (refr.has_value()) {
            auto wi = refr.value().normalized();
            auto sgeom = ShadingGeometry::make(normal, wi, wo);
            f32 transmittance = m_transmittance.eval_single(lambdas[0]);

            return BSDFSample{
                .bsdf = spectral::make_constant(1.f - fresnel_refl) * transmittance /
                        sgeom.cos_theta,
                .wi = wi,
                .pdf = 1.f - fresnel_refl,
                .did_refract = true,
            };
        } else {
            // Total internal reflection
            return reflect();
        }
    }
}
