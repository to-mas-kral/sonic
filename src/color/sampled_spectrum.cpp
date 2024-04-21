
#include "sampled_spectrum.h"
#include "spectrum.h"

#include <limits>

SampledSpectrum::
SampledSpectrum(const Array<f32, N_SPECTRUM_SAMPLES> &p_vals)
    : vals(p_vals) {}

SampledSpectrum
SampledSpectrum::make_constant(f32 constant) {
    SampledSpectrum sq{};
    sq.vals.fill(constant);
    return sq;
}

f32
SampledSpectrum::average() {
    f32 sum = 0.f;
    for (auto v : vals) {
        sum += v;
    }

    return sum / static_cast<f32>(N_SPECTRUM_SAMPLES);
}

f32
SampledSpectrum::max_component() const {
    f32 max = std::numeric_limits<f32>::min();
    for (f32 v : vals) {
        if (v > max) {
            max = v;
        }
    }

    return max;
}

void
SampledSpectrum::div_pdf(f32 pdf) {
    for (f32 &v : vals) {
        if (pdf != 0.f) {
            v /= pdf;
        }
    }
}

SampledSpectrum
SampledSpectrum::ONE() {
    SampledSpectrum sq{};
    sq.vals.fill(1.f);
    return sq;
}

SampledSpectrum
SampledSpectrum::ZERO() {
    SampledSpectrum sq{};
    sq.vals.fill(0.f);
    return sq;
}

SampledSpectrum
SampledSpectrum::operator+(const SampledSpectrum &other) const {
    SampledSpectrum sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq.vals[i] = vals[i] + other.vals[i];
    }

    return sq;
}

SampledSpectrum &
SampledSpectrum::operator+=(const SampledSpectrum &other) {
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        vals[i] += other.vals[i];
    }

    return *this;
}

SampledSpectrum
SampledSpectrum::operator-(const SampledSpectrum &other) const {
    SampledSpectrum sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq.vals[i] = vals[i] - other.vals[i];
    }

    return sq;
}

SampledSpectrum &
SampledSpectrum::operator-=(const SampledSpectrum &other) {
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        vals[i] -= other.vals[i];
    }

    return *this;
}

SampledSpectrum
SampledSpectrum::operator*(const SampledSpectrum &other) const {
    SampledSpectrum sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq.vals[i] = vals[i] * other.vals[i];
    }

    return sq;
}

SampledSpectrum &
SampledSpectrum::operator*=(const SampledSpectrum &other) {
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        vals[i] *= other.vals[i];
    }

    return *this;
}

SampledSpectrum
SampledSpectrum::operator*(f32 val) const {
    SampledSpectrum sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq.vals[i] = vals[i] * val;
    }

    return sq;
}

SampledSpectrum &
SampledSpectrum::operator*=(f32 val) {
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        vals[i] *= val;
    }

    return *this;
}

SampledSpectrum
SampledSpectrum::operator/(f32 div) const {
    SampledSpectrum sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq.vals[i] = vals[i] / div;
    }

    return sq;
}

SampledSpectrum
SampledSpectrum::operator/(const SampledSpectrum &other) const {
    SampledSpectrum sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq.vals[i] = vals[i] / other.vals[i];
    }

    return sq;
}

f32 &
SampledSpectrum::operator[](u32 index) {
    return vals[index];
}

SampledLambdas
SampledLambdas::new_sample_uniform(f32 rand) {
    SampledLambdas sl{};

    f32 lambda_min = static_cast<f32>(LAMBDA_MIN);
    f32 lambda_max = static_cast<f32>(LAMBDA_MAX);

    // Sample first wavelength
    sl.lambdas[0] = lerp(rand, lambda_min, lambda_max);

    if constexpr (N_SPECTRUM_SAMPLES > 1) {
        // Initialize remaining wavelenghts
        f32 delta = (lambda_max - lambda_min) / static_cast<f32>(N_SPECTRUM_SAMPLES);

        for (int i = 1; i < N_SPECTRUM_SAMPLES; i++) {
            sl.lambdas[i] = sl.lambdas[i - 1] + delta;
            if (sl.lambdas[i] > lambda_max) {
                sl.lambdas[i] = lambda_min + (sl.lambdas[i] - lambda_max);
            }
        }
    }

    return sl;
}

constexpr f32 PDF = 1.f / (static_cast<f32>(LAMBDA_MAX) - static_cast<f32>(LAMBDA_MIN));

vec3
SampledLambdas::to_xyz(const SampledSpectrum &radiance) {
    SampledSpectrum x = CIE_X.eval(*this) * radiance;
    SampledSpectrum y = CIE_Y.eval(*this) * radiance;
    SampledSpectrum z = CIE_Z.eval(*this) * radiance;

    x.div_pdf(PDF);
    y.div_pdf(PDF);
    z.div_pdf(PDF);

    f32 x_xyz = x.average() / CIE_Y_INTEGRAL;
    f32 y_xyz = y.average() / CIE_Y_INTEGRAL;
    f32 z_xyz = z.average() / CIE_Y_INTEGRAL;
    return vec3(x_xyz, y_xyz, z_xyz);
}

SampledLambdas
SampledLambdas::new_mock() {
    SampledLambdas sl{};
    sl.lambdas.fill(400.f);
    return sl;
}

const f32 &
SampledLambdas::operator[](u32 index) const {
    return lambdas[index];
}
