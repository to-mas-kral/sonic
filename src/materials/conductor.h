#ifndef PT_CONDUCTOR_H
#define PT_CONDUCTOR_H

#include "../integrator/shading_frame.h"
#include "../scene/texture.h"
#include "bsdf_sample.h"

struct ConductorMaterial {
    ConductorMaterial(const bool m_perfect, SpectrumTexture *const m_eta,
                      SpectrumTexture *const m_k)
        : m_perfect(m_perfect), m_eta(m_eta), m_k(m_k) {}

    static f32
    pdf();

    spectral
    eval(const ShadingFrame &sframe, const SampledLambdas &lambdas, const vec2 &uv) const;

    BSDFSample
    sample(const ShadingFrameIncomplete &sframe, const norm_vec3 &wo,
           const SampledLambdas &lambdas, const vec2 &uv) const;

private:
    // No Fresnel calculations, perfect reflector...
    bool m_perfect;
    // real part of the IOR
    SpectrumTexture *m_eta;
    // absorption coefficient
    SpectrumTexture *m_k;
};

#endif // PT_CONDUCTOR_H
