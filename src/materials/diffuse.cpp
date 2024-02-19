#include "diffuse.h"

#include "../math/sampling.h"

f32
DiffuseMaterial::pdf(const ShadingGeometry &sgeom) {
    return sgeom.cos_theta / M_PIf;
}

spectral
DiffuseMaterial::eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
                      const Texture *textures, const vec2 &uv) const {
    const Texture *texture = &textures[reflectance_tex_id];
    tuple3 refl_sigmoid_coeff = texture->fetch(uv);
    spectral refl = RgbSpectrum::from_coeff(refl_sigmoid_coeff).eval(lambdas);

    return refl / M_PIf;
}

BSDFSample
DiffuseMaterial::sample(const norm_vec3 &normal, const norm_vec3 &wo, const vec2 &sample,
                        const SampledLambdas &lambdas, const Texture *textures,
                        const vec2 &uv) const {
    norm_vec3 sample_dir = sample_cosine_hemisphere(sample);
    norm_vec3 wi = transform_frame(sample_dir, normal);
    auto sgeom = ShadingGeometry::make(normal, wi, wo);
    return BSDFSample{
        .bsdf = eval(sgeom, lambdas, textures, uv),
        .wi = wi,
        .pdf = DiffuseMaterial::pdf(sgeom),
        .did_refract = false,
    };
}
