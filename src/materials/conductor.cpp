#include "conductor.h"

#include "common.h"

#include <complex>

f32
ConductorMaterial::pdf() {
    return 0.F;
}

spectral
ConductorMaterial::eval(const ShadingFrame &sframe, const SampledLambdas &lambdas,
                        const vec2 &uv) const {
    if (m_perfect) {
        return spectral::ONE() / sframe.nowi();
    } else {
        // TODO: have to store the current IOR... when it isn't 1...
        auto rel_ior = m_eta->fetch(uv, lambdas);
        auto k = m_k->fetch(uv, lambdas);

        spectral fresnel = spectral::ZERO();
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            fresnel[i] =
                fresnel_conductor(std::complex<f32>(rel_ior[i], k[i]), sframe.howo());
        }
        return fresnel / sframe.nowi();
    }
}

BSDFSample
ConductorMaterial::sample(const ShadingFrameIncomplete &sframe, const norm_vec3 &wo,
                          const SampledLambdas &lambdas, const vec2 &uv) const {
    const norm_vec3 wi = ShadingFrameIncomplete::reflect(wo);
    const auto sframe_complete = ShadingFrame(sframe, wi, wo);

    return BSDFSample(eval(sframe_complete, lambdas, uv), 1.F, false, sframe_complete);
}
