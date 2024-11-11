
#include "sampled_spectrum.h"
#include "spectrum.h"

#include <algorithm>
#include <limits>

SampledSpectrum::
SampledSpectrum(const std::array<f32, N_SPECTRUM_SAMPLES> &p_vals)
    : vals(p_vals) {}

SampledSpectrum
SampledSpectrum::make_constant(const f32 constant) {
    SampledSpectrum sq{};
    sq.vals.fill(constant);
    return sq;
}

f32
SampledSpectrum::average() const {
    f32 sum = 0.F;
    for (const auto val : vals) {
        sum += val;
    }

    return sum / static_cast<f32>(N_SPECTRUM_SAMPLES);
}

f32
SampledSpectrum::max_component() const {
    f32 max = std::numeric_limits<f32>::min();
    for (const f32 val : vals) {
        max = std::max(val, max);
    }

    return max;
}

bool
SampledSpectrum::isnan() const {
    for (const auto &val : vals) {
        if (std::isnan(val)) {
            return true;
        }
    }

    return false;
}

bool
SampledSpectrum::isinf() const {
    for (const auto &val : vals) {
        if (std::isinf(val)) {
            return true;
        }
    }

    return false;
}

bool
SampledSpectrum::is_negative() const {
    for (const auto &val : vals) {
        if (val < 0.F) {
            return true;
        }
    }

    return false;
}

bool
SampledSpectrum::is_constant() const {
    const auto first = vals[0];
    for (int i = 1; i < N_SPECTRUM_SAMPLES; ++i) {
        if (vals[i] != first) {
            return false;
        }
    }

    return true;
}

bool
SampledSpectrum::is_zero() const {
    for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
        if (vals[i] != 0.F) {
            return false;
        }
    }

    return true;
}

void
SampledSpectrum::div_pdf(const f32 pdf) {
    for (f32 &val : vals) {
        if (pdf != 0.F) {
            val /= pdf;
        }
    }
}

std::string
SampledSpectrum::to_str() const {
    std::string str = "[";
    for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
        str += std::to_string(vals[i]);
        str += " ";
    }

    str += "]";

    return str;
}

void
SampledSpectrum::clamp(const f32 low, const f32 high) {
    for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
        vals[i] = std::clamp(vals[i], low, high);
    }
}

SampledSpectrum
SampledSpectrum::ONE() {
    SampledSpectrum sq{};
    sq.vals.fill(1.F);
    return sq;
}

SampledSpectrum
SampledSpectrum::ZERO() {
    SampledSpectrum sq{};
    sq.vals.fill(0.F);
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
SampledSpectrum::operator*(const f32 val) const {
    SampledSpectrum sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq.vals[i] = vals[i] * val;
    }

    return sq;
}

SampledSpectrum &
SampledSpectrum::operator*=(const f32 val) {
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        vals[i] *= val;
    }

    return *this;
}

SampledSpectrum
SampledSpectrum::operator/(const f32 div) const {
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
SampledSpectrum::operator[](const u32 index) {
    return vals[index];
}

const f32 &
SampledSpectrum::operator[](const u32 index) const {
    return vals[index];
}

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

static constexpr f32 PDF = 1.F / (static_cast<f32>(LAMBDA_MAX) - static_cast<f32>(LAMBDA_MIN));

vec3
SampledLambdas::to_xyz(const SampledSpectrum &radiance) const {
    auto rad = radiance;
    auto pdf = PDF;
    if (is_secondary_terminated) {
        for (int i = 1; i < N_SPECTRUM_SAMPLES; ++i) {
            rad[i] = 0.F;
        }
        pdf /= N_SPECTRUM_SAMPLES;
    }

    SampledSpectrum x = CIE_X.eval(*this) * rad;
    SampledSpectrum y = CIE_Y.eval(*this) * rad;
    SampledSpectrum z = CIE_Z.eval(*this) * rad;

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
