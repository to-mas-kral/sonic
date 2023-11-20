
#include "light_sampler.h"

LightSampler::LightSampler(const SharedVector<Light> &lights) {
    if (lights.size() == 0) {
        return;
    }

    has_lights = true;

    f32 total_power = 0.f;
    for (int l = 0; l < lights.size(); l++) {
        f32 power = lights[l].emitter.power();
        total_power += power;
    }

    auto num_lights = lights.size();
    pmf = SharedVector<f32>(num_lights);
    cmf = SharedVector<f32>(num_lights);

    for (int l = 0; l < lights.size(); l++) {
        f32 prob = lights[l].emitter.power() / total_power;
        pmf.push(std::move(prob));
    }

    f32 sum = 0.f;
    for (int i = 0; i < num_lights; i++) {
        sum += pmf[i];
        // TODO: do someting about this std::move nonsense
        cmf.push(sum + 0.f);
    }

    assert(cmf[cmf.size() - 1] == 1.);
}
