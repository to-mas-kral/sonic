#ifndef PT_ROUGH_PLASTIC_H
#define PT_ROUGH_PLASTIC_H

#include "../scene/texture.h"
#include "../spectrum/spectrum.h"
#include "../utils/basic_types.h"
#include "bsdf_sample.h"

class ShadingFrameIncomplete;
class ShadingFrame;

struct RoughCoatedDiffuseMaterial {
    RoughCoatedDiffuseMaterial(FloatTexture *const m_alpha, const Spectrum &ext_ior,
                               const Spectrum &int_ior,
                               SpectrumTexture *const diffuse_reflectance)
        : m_alpha(m_alpha), ext_ior(ext_ior), int_ior(int_ior),
          diffuse_reflectance(diffuse_reflectance) {}

    f32
    pdf(const ShadingFrame &sframe, const SampledLambdas &lambdas, const vec2 &uv) const;

    spectral
    eval(const ShadingFrame &sframe, const SampledLambdas &lambdas, const vec2 &uv) const;

    std::optional<BSDFSample>
    sample(const ShadingFrameIncomplete &sframe, const norm_vec3 &wo, const vec3 &xi,
           const SampledLambdas &lambdas, const vec2 &uv) const;

private:
    FloatTexture *m_alpha;
    Spectrum ext_ior;
    Spectrum int_ior;
    SpectrumTexture *diffuse_reflectance;
};

#endif // PT_ROUGH_PLASTIC_H
