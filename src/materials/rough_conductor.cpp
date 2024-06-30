#include "rough_conductor.h"
#include "common.h"

f32
RoughConductorMaterial::pdf(const ShadingFrame &sframe, const vec2 &uv) const {
    const auto alpha = fetch_alpha(m_alpha, uv);
    return TrowbridgeReitzGGX::pdf(sframe, alpha);
}

spectral
RoughConductorMaterial::eval(const ShadingFrame &sframe,
                             const SampledLambdas &lambdas, const vec2 &uv) const {
    // TODO: have to store the current IOR... when it isn't 1...
    auto rel_ior = m_eta->fetch(uv, lambdas);
    auto k = m_k->fetch(uv, lambdas);
    const auto alpha = fetch_alpha(m_alpha, uv);

    const f32 D = TrowbridgeReitzGGX::D(sframe.noh(), alpha);

    spectral fresnel = spectral::ZERO();
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        fresnel[i] =
            fresnel_conductor(std::complex<f32>(rel_ior[i], k[i]), sframe.howo());
    }

    const f32 G = TrowbridgeReitzGGX::G1(sframe.nowi(), sframe.howo(), alpha) *
                  TrowbridgeReitzGGX::G1(sframe.nowo(), sframe.howo(), alpha);
    return (fresnel * G * D) / (4.f * sframe.nowo() * sframe.nowi());

    // TODO: try the height-correlated smith
    /*f32 V = visibility_smith_height_correlated_ggx(sframe.nowo, sframe.nowi, alpha);
    return fresnel * V * D;*/
}

std::optional<BSDFSample>
RoughConductorMaterial::sample(const ShadingFrameIncomplete &sframe, const norm_vec3 &wo,
                               const vec2 &xi, const SampledLambdas &lambdas,
                               const vec2 &uv) const {
    const auto alpha = fetch_alpha(m_alpha, uv);

    const norm_vec3 wi = TrowbridgeReitzGGX::sample(wo, xi, alpha);

    if (!ShadingFrameIncomplete::same_hemisphere(wi, wo)) {
        return {};
    }

    const auto sframe_complete = ShadingFrame(sframe, wi, wo);

    return BSDFSample{
        .bsdf = eval(sframe_complete, lambdas, uv),
        .pdf = pdf(sframe_complete, uv),
        .did_refract = false,
        .sframe = sframe_complete,
    };
}
