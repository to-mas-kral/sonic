#include "sampled_lambdas.h"

#include "spectrum.h"
#include "spectrum_consts.h"

SampledLambdas
SampledLambdas::new_sample_uniform(const f32 rand) {
    SampledLambdas sl{};

    constexpr f32 lambda_min = static_cast<f32>(LAMBDA_MIN);
    constexpr f32 lambda_max = static_cast<f32>(LAMBDA_MAX);

    // Sample first wavelength
    sl.lambdas[0] = lerp(rand, lambda_min, lambda_max);

    if constexpr (N_SPECTRUM_SAMPLES > 1) {
        // Initialize remaining wavelenghts
        constexpr f32 delta =
            (lambda_max - lambda_min) / static_cast<f32>(N_SPECTRUM_SAMPLES);

        for (int i = 1; i < N_SPECTRUM_SAMPLES; i++) {
            sl.lambdas[i] = sl.lambdas[i - 1] + delta;
            if (sl.lambdas[i] > lambda_max) {
                sl.lambdas[i] = lambda_min + (sl.lambdas[i] - lambda_max);
            }
        }
    }

    return sl;
}

static constexpr f32 PDF =
    1.F / (static_cast<f32>(LAMBDA_MAX) - static_cast<f32>(LAMBDA_MIN));

vec3
SampledLambdas::to_xyz(const SpectralQuantity &radiance) const {
    auto rad = radiance;
    auto pdf = PDF;
    if (is_secondary_terminated) {
        for (int i = 1; i < N_SPECTRUM_SAMPLES; ++i) {
            rad[i] = 0.F;
        }
        pdf /= N_SPECTRUM_SAMPLES;
    }

    SpectralQuantity x = CIE_X.eval(*this) * rad;
    SpectralQuantity y = CIE_Y.eval(*this) * rad;
    SpectralQuantity z = CIE_Z.eval(*this) * rad;

    x.div_pdf(pdf);
    y.div_pdf(pdf);
    z.div_pdf(pdf);

    const f32 x_xyz = x.average() / CIE_Y_INTEGRAL;
    const f32 y_xyz = y.average() / CIE_Y_INTEGRAL;
    const f32 z_xyz = z.average() / CIE_Y_INTEGRAL;
    return {x_xyz, y_xyz, z_xyz};
}

void
SampledLambdas::terminate_secondary() {
    is_secondary_terminated = true;
}

SampledLambdas
SampledLambdas::new_mock() {
    SampledLambdas sl{};
    sl.lambdas.fill(400.F);
    return sl;
}

const f32 &
SampledLambdas::operator[](const u32 index) const {
    return lambdas[index];
}

#include "sampled_lambdas.h"
