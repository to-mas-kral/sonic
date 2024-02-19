
#include "light_sampler.h"

LightSampler::LightSampler(const std::vector<Light> &lights, const Geometry &geom) {
    if (lights.empty()) {
        return;
    }

    has_lights = true;

    f32 total_power = 0.f;
    for (auto light : lights) {
        f32 power = light.power(geom);
        total_power += power;
    }

    auto pmf = std::vector<f32>();
    pmf.reserve(lights.size());

    for (auto light : lights) {
        f32 prob = light.power(geom) / total_power;
        pmf.push_back(prob);
    }

    sampling_dist = PiecewiseDist1D(std::move(pmf));
}

Option<LightSample>
LightSampler::sample(const std::vector<Light> &lights, f32 sample) {
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

f32
LightSampler::light_sample_pdf(u32 light_id) const {
    return sampling_dist.pdf(light_id);
}
