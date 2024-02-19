#ifndef PT_ROUGH_PLASTIC_H
#define PT_ROUGH_PLASTIC_H

#include "../color/spectrum.h"
#include "../integrator/utils.h"
#include "../scene/texture.h"
#include "../utils/basic_types.h"
#include "bsdf_sample.h"

struct RoughPlasticMaterial {
    f32
    pdf(const ShadingGeometry &sgeom, const SampledLambdas &λ) const;

    spectral
    eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
         const Texture *textures, const vec2 &uv) const;

    Option<BSDFSample>
    sample(const norm_vec3 &normal, const norm_vec3 &ωo, const vec3 &ξ,
           const SampledLambdas &λ, const Texture *textures, const vec2 &uv) const;

    f32 alpha;
    Spectrum ext_ior;
    Spectrum int_ior;
    u32 diffuse_reflectance_id;
};

#endif // PT_ROUGH_PLASTIC_H
