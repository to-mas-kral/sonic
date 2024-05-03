#ifndef PT_CONDUCTOR_H
#define PT_CONDUCTOR_H

#include "../integrator/utils.h"
#include "../scene/texture.h"
#include "../scene/texture_id.h"
#include "bsdf_sample.h"

struct ConductorMaterial {
    static f32
    pdf();

    spectral
    eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
         const Texture *textures, const vec2 &uv) const;

    BSDFSample
    sample(const norm_vec3 &normal, const norm_vec3 &wo, const SampledLambdas &lambdas,
           const Texture *textures, const vec2 &uv) const;

    // No Fresnel calculations, perfect reflector...
    bool m_perfect;
    // real part of the IOR
    TextureId m_eta;
    // absorption coefficient
    TextureId m_k;
};

#endif // PT_CONDUCTOR_H
