#ifndef PT_ROUGH_CONDUCTOR_H
#define PT_ROUGH_CONDUCTOR_H

#include "../color/spectrum.h"
#include "../utils/basic_types.h"
#include "trowbridge_reitz_ggx.h"

#include <complex>

struct RoughConductorMaterial {
    f32
    pdf(const ShadingGeometry &sgeom) const {
        return TrowbridgeReitzGGX::pdf(sgeom, m_alpha);
    }

    spectral
    eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas) const {
        // TODO: have to store the current IOR... when it isn't 1...
        spectral rel_ior = m_eta.eval(lambdas);
        spectral k = m_k.eval(lambdas);

        f32 D = TrowbridgeReitzGGX::D(sgeom.noh, m_alpha);

        spectral fresnel = spectral::ZERO();
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            fresnel[i] =
                fresnel_conductor(std::complex<f32>(rel_ior[i], k[i]), sgeom.howo);
        }

        float G = TrowbridgeReitzGGX::G1(sgeom.nowi, sgeom.howo, m_alpha) *
                  TrowbridgeReitzGGX::G1(sgeom.nowo, sgeom.howo, m_alpha);
        return (fresnel * G * D) / (4.f * sgeom.nowo * sgeom.nowi);

        // TODO: try the height-correlated smith
        /*f32 V = visibility_smith_height_correlated_ggx(sgeom.nowo, sgeom.nowi, alpha);
        return fresnel * V * D;*/
    }

    Option<BSDFSample>
    sample(const norm_vec3 &normal, const norm_vec3 &wo, const vec2 &ξ,
           const SampledLambdas &lambdas, const Texture *textures, const vec2 &uv) const {
        norm_vec3 wi = TrowbridgeReitzGGX::sample(normal, wo, ξ, m_alpha);
        auto sgeom = ShadingGeometry::make(normal, wi, wo);

        if (sgeom.nowi * sgeom.nowo <= 0.f) {
            return {};
        }

        return BSDFSample{
            .bsdf = eval(sgeom, lambdas),
            .wi = wi,
            .pdf = pdf(sgeom),
            .did_refract = false,
        };
    }

    // real part of the IOR
    Spectrum m_eta;
    // absorption coefficient
    Spectrum m_k;
    f32 m_alpha;
};

#endif // PT_ROUGH_CONDUCTOR_H
