#include "sampled_lambdas.h"

#include "spectrum.h"
#include "spectrum_consts.h"

static constexpr f32 UNIFORM_PDF =
    1.F / (static_cast<f32>(LAMBDA_MAX) - static_cast<f32>(LAMBDA_MIN));

SampledLambdas
SampledLambdas::sample_uniform(const f32 xi) {
    SampledLambdas sl{};

    constexpr f32 lambda_min = static_cast<f32>(LAMBDA_MIN);
    constexpr f32 lambda_max = static_cast<f32>(LAMBDA_MAX);

    // Sample first wavelength
    sl.lambdas[0] = lerp(xi, lambda_min, lambda_max);

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

    sl.pdfs = SpectralQuantity(UNIFORM_PDF);

    return sl;
}

/// Due to: RADZISZEWSKI, Michal; BORYCZKO, Krzysztof a ALDA, Witold.
/// An improved technique for full spectral rendering. Václav Skala - UNION Agency, 2009.
/// PDF Code adapted from PBRT.
SampledLambdas
SampledLambdas::sample_visual_importance(const f32 xi) {
    SampledLambdas sl{};

    // PDF f(x) = 0.003939804229 / cosh^2(0.0072 (x - 538))
    // CDF p(x) = int_0^x f(a) da
    // sampling_routine s(x) = p(x)^-1
    for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
        // RADZISZEWSKI uses a quasi-monte-carlo sequence for generating the samples.
        // Here, the technique from PBRT is used where a xi is generated and the rest are
        // placed at equal distances in the 0-1 range.
        auto lambda_xi = xi + static_cast<f32>(i) / static_cast<f32>(N_SPECTRUM_SAMPLES);

        if (lambda_xi > 1.F) {
            lambda_xi -= 1.F;
        }

        sl.lambdas[i] =
            538.F - 138.888889F * std::atanhf(0.85691062F - 1.82750197F * lambda_xi);
        sl.pdfs[i] = pdf_visual_importance(sl.lambdas[i]);
    }

    return sl;
}

f32
SampledLambdas::pdf_visual_importance(const f32 lambda) {
    // integral of 1 / (cosh(0.0072(x-538))^2) from LAMBDA_MIN to LAMBDA_MAX.
    constexpr f32 NORM_CONSTANT = 0.003939804229F;
    return NORM_CONSTANT / sqr(std::coshf(0.0072F * (lambda - 538.F)));
}

vec3
SampledLambdas::to_xyz(const SpectralQuantity &radiance) const {
    auto rad = radiance;
    auto local_pdfs = pdfs;
    if (is_secondary_terminated) {
        for (int i = 1; i < N_SPECTRUM_SAMPLES; ++i) {
            rad[i] = 0.F;
        }
        local_pdfs /= N_SPECTRUM_SAMPLES;
    }

    if (weights[0] != -1.F) {
        rad *= weights;
    }

    SpectralQuantity x = CIE_X.eval(*this) * rad;
    SpectralQuantity y = CIE_Y.eval(*this) * rad;
    SpectralQuantity z = CIE_Z.eval(*this) * rad;

    x.div_pdf(local_pdfs);
    y.div_pdf(local_pdfs);
    z.div_pdf(local_pdfs);

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
