#ifndef PT_DIELECTRIC_H
#define PT_DIELECTRIC_H

#include "../integrator/shading_frame.h"
#include "../scene/texture.h"
#include "bsdf_sample.h"

struct DielectricMaterial {
    DielectricMaterial(SpectrumTexture *const m_int_ior, const Spectrum &m_ext_ior,
                       const Spectrum &m_transmittance)
        : m_int_ior(m_int_ior), m_ext_ior(m_ext_ior), m_transmittance(m_transmittance) {}

    static f32
    pdf();

    static spectral
    eval();

    BSDFSample
    sample(const ShadingFrameIncomplete &sframe, const norm_vec3 &wo, const vec2 &sample,
           SampledLambdas &lambdas, const vec2 &uv, bool is_frontfacing) const;

private:
    SpectrumTexture *m_int_ior{nullptr};
    Spectrum m_ext_ior;
    Spectrum m_transmittance;
};

#endif // PT_DIELECTRIC_H
