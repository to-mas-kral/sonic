
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
