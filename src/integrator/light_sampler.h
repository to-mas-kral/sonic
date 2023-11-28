#ifndef PT_LIGHT_SAMPLER_H
#define PT_LIGHT_SAMPLER_H

#include "../emitter.h"
#include "../geometry/geometry.h"
#include "../math/piecewise_dist.h"
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
    explicit LightSampler(const SharedVector<Light> &lights, const Geometry &geom);

    /// Sample lights according to power
    __device__ __forceinline__ COption<LightSample>
    sample(const SharedVector<Light> &lights, f32 sample) const {
        if (!has_lights) {
            return cuda::std::nullopt;
        }

        u32 light_index = sampling_dist.sample(sample);
        f32 pdf = sampling_dist.pdf(light_index);

        return LightSample{
            .pdf = pdf,
            .light = lights[light_index],
        };
    }

    /// The pdf of a light being sampled
    __device__ __forceinline__ f32
    light_sample_pdf(u32 light_id) {
        return sampling_dist.pdf(light_id);
    }

private:
    bool has_lights = false;
    PiecewiseDist1D sampling_dist;
};

#endif // PT_LIGHT_SAMPLER_H
