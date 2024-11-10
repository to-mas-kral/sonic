#ifndef DIFFUSETRANSMISSION_H
#define DIFFUSETRANSMISSION_H

#include "../scene/texture.h"
#include "bsdf_sample.h"

class ShadingFrame;
class ShadingFrameIncomplete;

struct DiffuseTransmissionMaterial {
    DiffuseTransmissionMaterial(SpectrumTexture *const reflectance,
                                SpectrumTexture *const transmittace, const f32 scale)
        : reflectance(reflectance), transmittace(transmittace), scale(scale) {}

    static f32
    pdf(const ShadingFrame &sframe);

    spectral
    eval(const SampledLambdas &lambdas, const vec2 &uv) const;

    BSDFSample
    sample(const ShadingFrameIncomplete &sframe, const norm_vec3 &wo, const vec2 &sample,
           const SampledLambdas &lambdas, const vec2 &uv) const;

private:
    SpectrumTexture *reflectance{nullptr};
    SpectrumTexture *transmittace{nullptr};
    f32 scale{1.f};
};

#endif // DIFFUSETRANSMISSION_H
