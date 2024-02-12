#ifndef PT_DIFFUSE_H
#define PT_DIFFUSE_H

#include "../utils/basic_types.h"

struct DiffuseMaterial {
    __host__ __device__ static f32
    pdf(const ShadingGeometry &sgeom) {
        return sgeom.cos_theta / M_PIf;
    }

    __device__ spectral
    eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
         const Texture *textures, const vec2 &uv) const {
        const Texture *texture = &textures[reflectance_tex_id];
        tuple3 refl_sigmoid_coeff = texture->fetch(uv);
        spectral refl = RgbSpectrum::from_coeff(refl_sigmoid_coeff).eval(lambdas);

        return refl / M_PIf;
    }

    __device__ BSDFSample
    sample(const norm_vec3 &normal, const norm_vec3 &wo, const vec2 &sample,
           const SampledLambdas &lambdas, const Texture *textures, const vec2 &uv) const {
        norm_vec3 sample_dir = sample_cosine_hemisphere(sample);
        norm_vec3 wi = orient_dir(sample_dir, normal);
        auto sgeom = ShadingGeometry::make(normal, wi, wo);
        return BSDFSample{
            .bsdf = eval(sgeom, lambdas, textures, uv),
            .wi = wi,
            .pdf = DiffuseMaterial::pdf(sgeom),
            .did_refract = false,
        };
    }

    u32 reflectance_tex_id;
};

#endif // PT_DIFFUSE_H
