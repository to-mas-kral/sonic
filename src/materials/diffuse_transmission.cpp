#include "diffuse_transmission.h"

#include "../integrator/shading_frame.h"
#include "../math/sampling.h"
#include "common.h"

f32
DiffuseTransmissionMaterial::pdf(const ShadingFrame &sframe) {
    return sframe.nowi() / M_PIf;
}

spectral
DiffuseTransmissionMaterial::eval(const SampledLambdas &lambdas, const vec2 &uv) const {
    const auto refl = fetch_reflectance(reflectance, uv, lambdas);
    return refl * scale / M_PIf;
}

BSDFSample
DiffuseTransmissionMaterial::sample(const ShadingFrameIncomplete &sframe,
                                    const norm_vec3 &wo, const vec2 &sample,
                                    const SampledLambdas &lambdas, const vec2 &uv) const {
    const norm_vec3 wi = sample_cosine_hemisphere(sample);
    const auto sframe_complete = ShadingFrame(sframe, wi, wo);

    return BSDFSample(eval(lambdas, uv), pdf(sframe_complete), false, sframe_complete);
}
