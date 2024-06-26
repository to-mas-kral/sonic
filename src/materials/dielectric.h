#ifndef PT_DIELECTRIC_H
#define PT_DIELECTRIC_H

#include "../integrator/shading_frame.h"
#include "../scene/texture.h"
#include "bsdf_sample.h"

struct DielectricMaterial {
    static f32
    pdf();

    static spectral
    eval();

    BSDFSample
    sample(const ShadingFrameIncomplete &sframe, const norm_vec3 &wo, const vec2 &sample,
           SampledLambdas &lambdas, const vec2 &uv, bool is_frontfacing) const;

    SpectrumTexture *m_int_ior;
    Spectrum m_ext_ior;
    Spectrum m_transmittance;
};

#endif // PT_DIELECTRIC_H
