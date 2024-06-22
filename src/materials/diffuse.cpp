#include "diffuse.h"

#include "../math/sampling.h"
#include "common.h"

f32
DiffuseMaterial::pdf(const ShadingGeometry &sgeom) {
    return sgeom.cos_theta / M_PIf;
}

spectral
DiffuseMaterial::eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
                      const vec2 &uv) const {
    const auto refl = fetch_reflectance(reflectance, uv, lambdas);
    return refl / M_PIf;
}

BSDFSample
DiffuseMaterial::sample(const norm_vec3 &normal, const norm_vec3 &wo, const vec2 &sample,
                        const SampledLambdas &lambdas, const vec2 &uv) const {
    const norm_vec3 sample_dir = sample_cosine_hemisphere(sample);
    const norm_vec3 wi = transform_frame(sample_dir, normal);
    const auto sgeom = ShadingGeometry::make(normal, wi, wo);
    return BSDFSample{
        .bsdf = eval(sgeom, lambdas, uv),
        .wi = wi,
        .pdf = DiffuseMaterial::pdf(sgeom),
        .did_refract = false,
    };
}
