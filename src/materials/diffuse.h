#ifndef PT_DIFFUSE_H
#define PT_DIFFUSE_H

#include "../integrator/utils.h"
#include "../scene/texture.h"
#include "../utils/basic_types.h"
#include "bsdf_sample.h"

struct DiffuseMaterial {
    static f32
    pdf(const ShadingGeometry &sgeom);

    spectral
    eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
         const Texture *textures, const vec2 &uv) const;

    BSDFSample
    sample(const norm_vec3 &normal, const norm_vec3 &wo, const vec2 &sample,
           const SampledLambdas &lambdas, const Texture *textures, const vec2 &uv) const;

    u32 reflectance_tex_id;
};

#endif // PT_DIFFUSE_H
