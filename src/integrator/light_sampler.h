#ifndef PT_LIGHT_SAMPLER_H
#define PT_LIGHT_SAMPLER_H

#include "../geometry/geometry.h"
#include "../math/piecewise_dist.h"
#include "../math/sampling.h"
#include "../scene/emitter.h"
#include "../scene/light.h"

struct LightSample {
    f32 pdf;
    Light light;
};

class LightSampler {
public:
    LightSampler() = default;
    explicit LightSampler(const std::vector<Light> &lights, const Geometry &geom);

    /// Sample lights according to power
    Option<LightSample>
    sample(const std::vector<Light> &lights, f32 sample) {
        if (!has_lights) {
            return {};
        }

        u32 light_index = sampling_dist.sample(sample);
        f32 pdf = sampling_dist.pdf(light_index);

        return LightSample{
            .pdf = pdf,
            .light = lights[light_index],
        };
    }

    /// The pdf of a light being sampled
    f32
    light_sample_pdf(u32 light_id) const {
        return sampling_dist.pdf(light_id);
    }

private:
    bool has_lights = false;
    PiecewiseDist1D sampling_dist;
};

#endif // PT_LIGHT_SAMPLER_H
