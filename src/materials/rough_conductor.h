#ifndef PT_ROUGH_CONDUCTOR_H
#define PT_ROUGH_CONDUCTOR_H

#include "../color/spectrum.h"
#include "../integrator/utils.h"
#include "../scene/texture.h"
#include "../scene/texture_id.h"
#include "../utils/basic_types.h"
#include "trowbridge_reitz_ggx.h"

#include <complex>

struct RoughConductorMaterial {
    f32
    pdf(const ShadingGeometry &sgeom, const Texture *textures, const vec2 &uv) const;

    spectral
    eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
         const Texture *textures, const vec2 &uv) const;

    Option<BSDFSample>
    sample(const norm_vec3 &normal, const norm_vec3 &wo, const vec2 &ξ,
           const SampledLambdas &lambdas, const Texture *textures, const vec2 &uv) const;

    // real part of the IOR
    TextureId m_eta;
    // absorption coefficient
    TextureId m_k;
    TextureId m_alpha;
};

#endif // PT_ROUGH_CONDUCTOR_H
