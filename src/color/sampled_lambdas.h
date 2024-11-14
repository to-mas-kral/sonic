#ifndef SAMPLEDLAMBDAS_H
#define SAMPLEDLAMBDAS_H

#include "spectral_quantity.h"

class SampledLambdas {
public:
    static SampledLambdas
    new_sample_uniform(f32 rand);

    static SampledLambdas
    new_mock();

    vec3
    to_xyz(const SpectralQuantity &radiance) const;

    void
    terminate_secondary();

    const f32 &
    operator[](u32 index) const;

private:
    bool is_secondary_terminated{false};
    std::array<f32, N_SPECTRUM_SAMPLES> lambdas{};
};

#endif // SAMPLEDLAMBDAS_H
