#include "conductor.h"

#include "common.h"

#include <complex>

f32
ConductorMaterial::pdf() {
    return 0.f;
}

spectral
ConductorMaterial::eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
                        const Texture *textures, const vec2 &uv) const {
    if (m_perfect) {
        return spectral::ONE() / sgeom.cos_theta;
    } else {
        // TODO: have to store the current IOR... when it isn't 1...
        spectral rel_ior = m_eta.eval(lambdas);
        spectral k = m_k.eval(lambdas);

        spectral fresnel = spectral::ZERO();
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            fresnel[i] =
                fresnel_conductor(std::complex<f32>(rel_ior[i], k[i]), sgeom.howo);
        }
        return fresnel / sgeom.cos_theta;
    }
}

BSDFSample
ConductorMaterial::sample(const norm_vec3 &normal, const norm_vec3 &wo,
                          const SampledLambdas &lambdas, const Texture *textures,
                          const vec2 &uv) const {
    norm_vec3 wi = vec3::reflect(wo, normal).normalized();
    auto sgeom = ShadingGeometry::make(normal, wi, wo);
    return BSDFSample{
        .bsdf = eval(sgeom, lambdas, textures, uv),
        .wi = wi,
        .pdf = 1.f,
        .did_refract = false,
    };
}
