#ifndef PT_LIGHT_SAMPLER_H
#define PT_LIGHT_SAMPLER_H

#include "../geometry/geometry_container.h"
#include "../math/discrete_dist.h"
#include "../scene/light.h"

class Envmap;

struct LightIndexSample {
    f32 pdf;
    const Light *light;
};

class LightSampler {
public:
    LightSampler() = default;

    explicit
    LightSampler(const std::vector<Light> &lights, const GeometryContainer &geom);

    /// Sample lights according to power
    std::optional<LightIndexSample>
    sample(const std::vector<Light> &lights, f32 sample) const;

    /// The pdf of a light being sampled
    f32
    light_sample_pdf(u32 light_id) const;

private:
    std::optional<DiscreteDist> sampling_dist;
};

#endif // PT_LIGHT_SAMPLER_H
