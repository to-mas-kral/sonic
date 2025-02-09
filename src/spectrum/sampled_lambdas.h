#ifndef SAMPLEDLAMBDAS_H
#define SAMPLEDLAMBDAS_H

#include "../math/samplers/sampler.h"
#include "spectral_quantity.h"

class SampledLambdas {
public:
    static SampledLambdas
    new_sample_uniform(f32 xi);

    static SampledLambdas
    new_sample_importance(Sampler &sampler);

    static SampledLambdas
    new_mock();

    SampledLambdas(const std::array<f32, N_SPECTRUM_SAMPLES> &lambdas,
                   const spectral &pdfs)
        : pdfs(pdfs), lambdas(lambdas) {}

    SampledLambdas() {}

    vec3
    to_xyz(const SpectralQuantity &radiance) const;

    void
    terminate_secondary();

    const f32 &
    operator[](u32 index) const;

    spectral pdfs{};
    spectral weights{-1.F};

private:
    bool is_secondary_terminated{false};
    std::array<f32, N_SPECTRUM_SAMPLES> lambdas{};
};

#endif // SAMPLEDLAMBDAS_H
