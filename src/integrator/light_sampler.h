#ifndef PT_LIGHT_SAMPLER_H
#define PT_LIGHT_SAMPLER_H

#include "../emitter.h"
#include "../geometry/geometry.h"
#include "../math/sampling.h"
#include "../scene/light.h"

#include <cuda/std/optional>
#include <cuda/std/span>

struct LightSample {
    f32 pdf;
    Light light;
};

class LightSampler {
public:
    LightSampler() = default;
    explicit LightSampler(const SharedVector<Light> &lights);

    /// Sample lights according to power
    __device__ __forceinline__ COption<LightSample>
    sample(const SharedVector<Light> &lights, f32 sample) const {
        if (!has_lights) {
            return cuda::std::nullopt;
        }

        u32 light_index =
            sample_discrete_cmf(cuda::std::span(cmf.get_ptr(), cmf.size()), sample);

        f32 pdf = pmf[light_index];

        return LightSample{
            .pdf = pdf,
            .light = lights[light_index],
        };
    }

    /// The pdf of a light being sampled
    __device__ __forceinline__ f32
    light_sample_pdf(u32 light_id) {
        return pmf[light_id];
    }

private:
    bool has_lights = false;
    SharedVector<f32> pmf{}; // probability of sampling each light source
    SharedVector<f32> cmf{}; // cumulative function ^
};

#endif // PT_LIGHT_SAMPLER_H
