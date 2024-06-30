#ifndef PT_DIFFUSE_H
#define PT_DIFFUSE_H

#include "../scene/texture.h"
#include "../utils/basic_types.h"
#include "bsdf_sample.h"

class ShadingFrame;
class ShadingFrameIncomplete;

struct DiffuseMaterial {
    static f32
    pdf(const ShadingFrame &sframe);

    spectral
    eval(const SampledLambdas &lambdas, const vec2 &uv) const;

    BSDFSample
    sample(const ShadingFrameIncomplete &sframe, const norm_vec3 &wo, const vec2 &sample,
           const SampledLambdas &lambdas, const vec2 &uv) const;

    SpectrumTexture *reflectance;
};

#endif // PT_DIFFUSE_H
