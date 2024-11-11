
#include "light_sampler.h"

LightSampler::
LightSampler(const std::vector<Light> &lights, const Geometry &geom) {
    if (lights.empty()) {
        return;
    }

    has_lights = true;

    f32 total_power = 0.F;
    for (auto light : lights) {
        const f32 power = light.power(geom);
        total_power += power;
    }

    auto pmf = std::vector<f32>();
    pmf.reserve(lights.size());

    for (auto light : lights) {
        const f32 prob = light.power(geom) / total_power;
        pmf.push_back(prob);
    }

    sampling_dist = DiscreteDist(std::move(pmf));
}

std::optional<LightIndexSample>
LightSampler::sample(const std::vector<Light> &lights, const f32 sample) const {
    if (!has_lights) {
        return {};
    }

    const u32 light_index = sampling_dist.sample(sample);
    const f32 pdf = sampling_dist.pdf(light_index);

    return LightIndexSample{
        .pdf = pdf,
        .light = &lights[light_index],
    };
}

f32
LightSampler::light_sample_pdf(const u32 light_id) const {
    return sampling_dist.pdf(light_id);
}
