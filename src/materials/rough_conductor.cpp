#include "rough_conductor.h"
#include "common.h"

f32
RoughConductorMaterial::pdf(const ShadingGeometry &sgeom, const Texture *textures,
                            const vec2 &uv) const {
    const auto alpha = fetch_alpha(textures, m_alpha, uv);
    return TrowbridgeReitzGGX::pdf(sgeom, alpha);
}

spectral
RoughConductorMaterial::eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
                             const Texture *textures, const vec2 &uv) const {
    // TODO: have to store the current IOR... when it isn't 1...
    auto eta_spec = textures[m_eta.inner].fetch_spectrum(uv);
    auto k_spec = textures[m_k.inner].fetch_spectrum(uv);
    const auto alpha = fetch_alpha(textures, m_alpha, uv);

    spectral rel_ior = eta_spec.eval(lambdas);
    spectral k = k_spec.eval(lambdas);

    f32 D = TrowbridgeReitzGGX::D(sgeom.noh, alpha);

    spectral fresnel = spectral::ZERO();
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        fresnel[i] = fresnel_conductor(std::complex<f32>(rel_ior[i], k[i]), sgeom.howo);
    }

    float G = TrowbridgeReitzGGX::G1(sgeom.nowi, sgeom.howo, alpha) *
              TrowbridgeReitzGGX::G1(sgeom.nowo, sgeom.howo, alpha);
    return (fresnel * G * D) / (4.f * sgeom.nowo * sgeom.nowi);

    // TODO: try the height-correlated smith
    /*f32 V = visibility_smith_height_correlated_ggx(sgeom.nowo, sgeom.nowi, alpha);
    return fresnel * V * D;*/
}

Option<BSDFSample>
RoughConductorMaterial::sample(const norm_vec3 &normal, const norm_vec3 &wo,
                               const vec2 &ξ, const SampledLambdas &lambdas,
                               const Texture *textures, const vec2 &uv) const {
    const auto alpha = fetch_alpha(textures, m_alpha, uv);

    norm_vec3 wi = TrowbridgeReitzGGX::sample(normal, wo, ξ, alpha);
    auto sgeom = ShadingGeometry::make(normal, wi, wo);

    if (sgeom.nowi * sgeom.nowo <= 0.f) {
        return {};
    }

    return BSDFSample{
        .bsdf = eval(sgeom, lambdas, textures, uv),
        .wi = wi,
        .pdf = pdf(sgeom, textures, uv),
        .did_refract = false,
    };
}
