#ifndef PT_ROUGH_CONDUCTOR_H
#define PT_ROUGH_CONDUCTOR_H

#include "../scene/texture.h"
#include "../utils/basic_types.h"
#include "trowbridge_reitz_ggx.h"

struct RoughConductorMaterial {
    f32
    pdf(const ShadingFrame &sframe, const vec2 &uv) const;

    spectral
    eval(const ShadingFrame &sframe, const SampledLambdas &lambdas, const vec2 &uv) const;

    std::optional<BSDFSample>
    sample(const ShadingFrameIncomplete &sframe, const norm_vec3 &wo, const vec2 &xi,
           const SampledLambdas &lambdas, const vec2 &uv) const;

    // real part of the IOR
    SpectrumTexture *m_eta;
    // absorption coefficient
    SpectrumTexture *m_k;
    FloatTexture *m_alpha;
};

#endif // PT_ROUGH_CONDUCTOR_H
