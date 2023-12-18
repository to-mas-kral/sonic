
#include "light_sampler.h"

LightSampler::LightSampler(const UmVector<Light> &lights, const Geometry &geom) {
    if (lights.size() == 0) {
        return;
    }

    has_lights = true;

    f32 total_power = 0.f;
    for (int l = 0; l < lights.size(); l++) {
        f32 power = lights[l].power(geom);
        total_power += power;
    }

    auto num_lights = lights.size();
    auto pmf = UmVector<f32>(num_lights);

    for (int l = 0; l < lights.size(); l++) {
        f32 prob = lights[l].power(geom) / total_power;
        pmf.push(prob);
    }

    sampling_dist = PiecewiseDist1D(std::move(pmf));
}
