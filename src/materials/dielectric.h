#ifndef PT_DIELECTRIC_H
#define PT_DIELECTRIC_H

#include "../scene/texture.h"
#include "../scene/texture_id.h"
#include "bsdf_sample.h"

struct DielectricMaterial {
    static f32
    pdf();

    static spectral
    eval();

    BSDFSample
    sample(const norm_vec3 &normal, const norm_vec3 &wo, const vec2 &sample,
           const SampledLambdas &lambdas, const Texture *textures, const vec2 &uv,
           bool is_frontfacing) const;

    TextureId m_int_ior;
    Spectrum m_ext_ior;
    Spectrum m_transmittance;
};

#endif // PT_DIELECTRIC_H
