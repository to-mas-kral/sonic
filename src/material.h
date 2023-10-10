#ifndef PT_MATERIAL_H
#define PT_MATERIAL_H

#include "math/sampling.h"
#include "math/vecmath.h"
#include "utils/numtypes.h"
#include "utils/rng.h"

// TODO: Probably make into a std::variant or something ?
class Material {
public:
    explicit Material(const vec3 &reflectance) : reflectance(reflectance) {}

    // TODO: currently only sample, pdf, eval functions are only for diffuse BRDFs
    // now...

    __device__ vec3 sample(vec3 normal, vec3 view_dir, curandState *rand_state) const {
        vec3 sample_dir = sample_cosine_hemisphere(rand_state);
        return orient_dir(sample_dir, normal);
    }

    // Probability density function of sampling the BRDF
    __device__ f32 pdf(f32 cos_theta) const { return cos_theta / M_PIf; }

    __device__ vec3 eval() const { return reflectance / M_PIf; }

    vec3 reflectance;
};

#endif // PT_MATERIAL_H
