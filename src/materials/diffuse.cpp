#include "diffuse.h"

#include "../integrator/shading_frame.h"
#include "../math/sampling.h"
#include "common.h"

f32
DiffuseMaterial::pdf(const ShadingFrame &sframe) {
    return sframe.nowi() / M_PIf;
}

spectral
DiffuseMaterial::eval(const SampledLambdas &lambdas, const vec2 &uv) const {
    const auto refl = fetch_reflectance(reflectance, uv, lambdas);
    return refl / M_PIf;
}

BSDFSample
DiffuseMaterial::sample(const ShadingFrameIncomplete &sframe, const norm_vec3 &wo,
                        const vec2 &sample, const SampledLambdas &lambdas,
                        const vec2 &uv) const {
    const norm_vec3 wi = sample_cosine_hemisphere(sample);

    const auto sframe_complete = ShadingFrame(sframe, wi, wo);
    return BSDFSample(eval(lambdas, uv), pdf(sframe_complete), false, sframe_complete);
}
