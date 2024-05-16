#ifndef PT_ROUGH_PLASTIC_H
#define PT_ROUGH_PLASTIC_H

#include "../color/spectrum.h"
#include "../integrator/utils.h"
#include "../scene/texture.h"
#include "../utils/basic_types.h"
#include "bsdf_sample.h"

struct RoughPlasticMaterial {
    f32
    pdf(const ShadingGeometry &sgeom, const SampledLambdas &λ, const vec2 &uv) const;

    spectral
    eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
         const vec2 &uv) const;

    Option<BSDFSample>
    sample(const norm_vec3 &normal, const norm_vec3 &ωo, const vec3 &ξ,
           const SampledLambdas &λ, const vec2 &uv) const;

    FloatTexture *m_alpha;
    Spectrum ext_ior;
    Spectrum int_ior;
    SpectrumTexture *diffuse_reflectance;
};

#endif // PT_ROUGH_PLASTIC_H
