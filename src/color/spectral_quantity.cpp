
#include "spectral_quantity.h"
#include "spectrum.h"

#include <algorithm>
#include <limits>

SpectralQuantity::SpectralQuantity(const std::array<f32, N_SPECTRUM_SAMPLES> &p_vals)
    : vals(p_vals) {}

SpectralQuantity::SpectralQuantity(const f32 val) { vals.fill(val); }

SpectralQuantity
SpectralQuantity::make_constant(const f32 constant) {
    SpectralQuantity sq{};
    sq.vals.fill(constant);
    return sq;
}

f32
SpectralQuantity::average() const {
    f32 sum = 0.F;
    for (const auto val : vals) {
        sum += val;
    }

    return sum / static_cast<f32>(N_SPECTRUM_SAMPLES);
}

f32
SpectralQuantity::max_component() const {
    f32 max = std::numeric_limits<f32>::min();
    for (const f32 val : vals) {
        max = std::max(val, max);
    }

    return max;
}

bool
SpectralQuantity::isnan() const {
    for (const auto &val : vals) {
        if (std::isnan(val)) {
            return true;
        }
    }

    return false;
}

bool
SpectralQuantity::isinf() const {
    for (const auto &val : vals) {
        if (std::isinf(val)) {
            return true;
        }
    }

    return false;
}

bool
SpectralQuantity::is_negative() const {
    for (const auto &val : vals) {
        if (val < 0.F) {
            return true;
        }
    }

    return false;
}

bool
SpectralQuantity::is_constant() const {
    const auto first = vals[0];
    for (int i = 1; i < N_SPECTRUM_SAMPLES; ++i) {
        if (vals[i] != first) {
            return false;
        }
    }

    return true;
}

bool
SpectralQuantity::is_zero() const {
    for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
        if (vals[i] != 0.F) {
            return false;
        }
    }

    return true;
}

void
SpectralQuantity::div_pdf(const SpectralQuantity &pdf) {
    for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
        if (pdf[i] != 0.F) {
            vals[i] /= pdf[i];
        }
    }
}

std::string
SpectralQuantity::to_str() const {
    std::string str = "[";
    for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
        str += std::to_string(vals[i]);
        str += " ";
    }

    str += "]";

    return str;
}

void
SpectralQuantity::clamp(const f32 low, const f32 high) {
    for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
        vals[i] = std::clamp(vals[i], low, high);
    }
}

SpectralQuantity
SpectralQuantity::ONE() {
    SpectralQuantity sq{};
    sq.vals.fill(1.F);
    return sq;
}

SpectralQuantity
SpectralQuantity::ZERO() {
    SpectralQuantity sq{};
    sq.vals.fill(0.F);
    return sq;
}

SpectralQuantity
SpectralQuantity::operator+(const SpectralQuantity &other) const {
    SpectralQuantity sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq.vals[i] = vals[i] + other.vals[i];
    }

    return sq;
}

SpectralQuantity &
SpectralQuantity::operator+=(const SpectralQuantity &other) {
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        vals[i] += other.vals[i];
    }

    return *this;
}

SpectralQuantity
SpectralQuantity::operator-(const SpectralQuantity &other) const {
    SpectralQuantity sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq.vals[i] = vals[i] - other.vals[i];
    }

    return sq;
}

SpectralQuantity &
SpectralQuantity::operator-=(const SpectralQuantity &other) {
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        vals[i] -= other.vals[i];
    }

    return *this;
}

SpectralQuantity
SpectralQuantity::operator*(const SpectralQuantity &other) const {
    SpectralQuantity sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq.vals[i] = vals[i] * other.vals[i];
    }

    return sq;
}

SpectralQuantity &
SpectralQuantity::operator*=(const SpectralQuantity &other) {
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        vals[i] *= other.vals[i];
    }

    return *this;
}

SpectralQuantity
SpectralQuantity::operator*(const f32 val) const {
    SpectralQuantity sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq.vals[i] = vals[i] * val;
    }

    return sq;
}

SpectralQuantity &
SpectralQuantity::operator*=(const f32 val) {
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        vals[i] *= val;
    }

    return *this;
}

SpectralQuantity
SpectralQuantity::operator/(const f32 div) const {
    SpectralQuantity sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq.vals[i] = vals[i] / div;
    }

    return sq;
}

SpectralQuantity
SpectralQuantity::operator/(const SpectralQuantity &other) const {
    SpectralQuantity sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq.vals[i] = vals[i] / other.vals[i];
    }

    return sq;
}

SpectralQuantity &
SpectralQuantity::operator/=(const f32 div) {
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        vals[i] /= div;
    }

    return *this;
}

f32 &
SpectralQuantity::operator[](const u32 index) {
    return vals[index];
}

const f32 &
SpectralQuantity::operator[](const u32 index) const {
    return vals[index];
}
