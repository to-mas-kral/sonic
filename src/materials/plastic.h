#ifndef PT_PLASTIC_H
#define PT_PLASTIC_H

#include "../integrator/utils.h"
#include "../scene/texture.h"
#include "../scene/texture_id.h"
#include "../utils/basic_types.h"
#include "bsdf_sample.h"

/// Math taken from:
/// Physically Based Specular + Diffuse
/// Jan van Bergen
struct PlasticMaterial {
    f32
    pdf(const ShadingGeometry &sgeom, const SampledLambdas &λ) const;

    static bool
    is_dirac_delta();

    spectral
    eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
         const Texture *textures, const vec2 &uv) const;

    BSDFSample
    sample(const norm_vec3 &normal, const norm_vec3 &ωo, const vec3 &ξ,
           const SampledLambdas &λ, const Texture *textures, const vec2 &uv) const;

    Spectrum ext_ior;
    Spectrum int_ior;
    TextureId diffuse_reflectance_id;
};

#endif // PT_PLASTIC_H
