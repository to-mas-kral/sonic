#include "rough_coateddiffuse.h"

#include "../integrator/shading_frame.h"
#include "../math/sampling.h"
#include "common.h"
#include "trowbridge_reitz_ggx.h"

f32
RoughCoatedDiffuseMaterial::pdf(const ShadingFrame &sframe,
                          const SampledLambdas &lambdas, const vec2 &uv) const {
    const f32 int_η = int_ior.eval_single(lambdas[0]);
    const f32 ext_η = ext_ior.eval_single(lambdas[0]);
    const f32 rel_η = int_η / ext_η;

    const f32 fresnel_i = fresnel_dielectric(rel_η, sframe.nowo());

    const f32 diffuse_pdf = (1.f - fresnel_i) * sframe.nowi() / M_PIf;
    const f32 alpha = fetch_alpha(m_alpha, uv);
    const f32 microfacet_pdf = fresnel_i * TrowbridgeReitzGGX::pdf(sframe, alpha);

    return diffuse_pdf + microfacet_pdf;
}

spectral
RoughCoatedDiffuseMaterial::eval(const ShadingFrame &sframe,
                           const SampledLambdas &lambdas, const vec2 &uv) const {
    const f32 int_ior_s = int_ior.eval_single(lambdas[0]);
    const f32 ext_ior_s = ext_ior.eval_single(lambdas[0]);
    /// This is external / internal !
    const f32 rel_ior = ext_ior_s / int_ior_s;

    const f32 fresnel_i = fresnel_dielectric(1.f / rel_ior, sframe.nowi());

    // Specular case
    const f32 alpha = fetch_alpha(m_alpha, uv);
    const f32 D = TrowbridgeReitzGGX::D(sframe.noh(), alpha);

    const f32 G = TrowbridgeReitzGGX::G1(sframe.nowi(), sframe.howo(), alpha) *
                  TrowbridgeReitzGGX::G1(sframe.nowo(), sframe.howo(), alpha);
    const spectral microfacet_brdf = spectral::make_constant(fresnel_i * G * D) /
                                     (4.f * sframe.nowo() * sframe.nowi());

    // TODO: try the height-correlated smith
    /*const f32 V = visibility_smith_height_correlated_ggx(sgeom.nowo, sgeom.nowi,
    alpha); return fresnel * V * D;*/

    // Figure out the angle of the ray that refracts from inside to outisde wo
    const f32 cos_theta_out = sframe.nowo();
    const f32 sin_theta_out = safe_sqrt(1.f - sqr(cos_theta_out));
    const f32 sin_theta_in = sin_theta_out * rel_ior;
    const f32 cos_theta_in = safe_sqrt(1.f - sqr(sin_theta_in));
    const f32 fresnel_o = fresnel_dielectric(rel_ior, cos_theta_in);

    const auto α = fetch_reflectance(diffuse_reflectance, uv, lambdas);

    f32 re = 0.919317f;
    f32 ior_pow = int_ior_s;
    re -= (3.4793f / ior_pow);
    ior_pow *= ior_pow;
    re += (6.75335f / ior_pow);
    ior_pow *= ior_pow;
    re -= (7.80989f / ior_pow);
    ior_pow *= ior_pow;
    re += (4.98554f / ior_pow);
    ior_pow *= ior_pow;
    re -= (1.36881f / ior_pow);

    const f32 ri = 1.f - sqr(rel_ior) * (1.f - re);

    const spectral scattering_brdf = α * (1.f - fresnel_i) * (1.f - fresnel_o) /
                                     ((spectral::make_constant(1.f) - α * ri) * M_PIf);

    const spectral brdf = microfacet_brdf + scattering_brdf * sqr(rel_ior);

    return brdf;
}

std::optional<BSDFSample>
RoughCoatedDiffuseMaterial::sample(const ShadingFrameIncomplete &sframe, const norm_vec3 &wo,
                             const vec3 &xi, const SampledLambdas &lambdas,
                             const vec2 &uv) const {
    const f32 int_η = int_ior.eval_single(lambdas[0]);
    const f32 ext_η = ext_ior.eval_single(lambdas[0]);
    const f32 rel_η = int_η / ext_η;

    const f32 nowo = ShadingFrameIncomplete::cos_theta(wo);
    const f32 potential_fresnel = fresnel_dielectric(rel_η, nowo);

    if (xi.z < potential_fresnel) {
        // sample specular
        const f32 alpha = fetch_alpha(m_alpha, uv);
        const norm_vec3 wi = TrowbridgeReitzGGX::sample(wo, vec2(xi.x, xi.y), alpha);

        if (!ShadingFrame::same_hemisphere(wi, wo)) {
            return {};
        }

        const auto sframe_complete = ShadingFrame(sframe, wi, wo);

        return BSDFSample{
            .bsdf = eval(sframe_complete, lambdas, uv),
            .pdf = pdf(sframe_complete, lambdas, uv),
            .sframe = sframe_complete,
        };
    } else {
        // sample diffuse
        const norm_vec3 wi = sample_cosine_hemisphere(vec2(xi.x, xi.y));
        const auto sframe_complete = ShadingFrame(sframe, wi, wo);

        if (sframe_complete.is_degenerate()) {
            return {};
        }

        return BSDFSample{
            .bsdf = eval(sframe_complete, lambdas, uv),
            .pdf = pdf(sframe_complete, lambdas, uv),
            .sframe = sframe_complete,
        };
    }
}
