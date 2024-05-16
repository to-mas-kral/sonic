#ifndef DIFFUSETRANSMISSION_H
#define DIFFUSETRANSMISSION_H

#include "../integrator/utils.h"
#include "../scene/texture.h"
#include "bsdf_sample.h"

struct DiffuseTransmissionMaterial {
    static f32
    pdf(const ShadingGeometry &sgeom);

    spectral
    eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
         const vec2 &uv) const;

    BSDFSample
    sample(const norm_vec3 &normal, const norm_vec3 &wo, const vec2 &sample,
           const SampledLambdas &lambdas, const vec2 &uv) const;

    SpectrumTexture *reflectance{nullptr};
    SpectrumTexture *transmittace{nullptr};
    f32 scale{1.f};
};

#endif // DIFFUSETRANSMISSION_H
