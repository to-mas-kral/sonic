#ifndef PT_PLASTIC_H
#define PT_PLASTIC_H

#include "../scene/texture.h"
#include "../utils/basic_types.h"
#include "bsdf_sample.h"

class ShadingFrame;
class ShadingFrameIncomplete;

/// Math taken from:
/// Physically Based Specular + Diffuse
/// Jan van Bergen
struct CoatedDifuseMaterial {
    f32
    pdf(const ShadingFrame &sframe, const SampledLambdas &lambdas) const;

    static bool
    is_dirac_delta();

    spectral
    eval(const ShadingFrame &sframe, const SampledLambdas &lambdas,
         const vec2 &uv) const;

    BSDFSample
    sample(const ShadingFrameIncomplete &sframe, const norm_vec3 &wo, const vec3 &xi,
           const SampledLambdas &lambdas, const vec2 &uv) const;

    Spectrum ext_ior;
    Spectrum int_ior;
    SpectrumTexture *diffuse_reflectance;
};

#endif // PT_PLASTIC_H
