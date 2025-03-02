#ifndef PT_SPECTRAL_QUANTITY_H
#define PT_SPECTRAL_QUANTITY_H

#include "../math/vecmath.h"
#include "../utils/basic_types.h"

#include <array>
#include <string>

constexpr u32 N_SPECTRUM_SAMPLES = 8;

class SpectralQuantity {
public:
    SpectralQuantity() = default;

    explicit SpectralQuantity(const std::array<f32, N_SPECTRUM_SAMPLES> &p_vals);

    explicit SpectralQuantity(f32 val);

    static SpectralQuantity
    make_constant(f32 constant);

    f32
    average() const;

    f32
    max_component() const;

    bool
    isnan() const;

    bool
    isinf() const;

    bool
    is_negative() const;

    bool
    is_constant() const;

    bool
    is_zero() const;

    bool
    is_invalid() const {
        return is_negative() || isnan() || isinf();
    }

    void
    div_pdf(const SpectralQuantity &pdf);

    std::string
    to_str() const;

    void
    clamp(f32 low, f32 high);

    static SpectralQuantity
    ONE();

    static SpectralQuantity
    ZERO();

    SpectralQuantity
    operator+(const SpectralQuantity &other) const;

    SpectralQuantity &
    operator+=(const SpectralQuantity &other);

    SpectralQuantity
    operator-(const SpectralQuantity &other) const;

    SpectralQuantity &
    operator-=(const SpectralQuantity &other);

    SpectralQuantity
    operator*(const SpectralQuantity &other) const;

    SpectralQuantity &
    operator*=(const SpectralQuantity &other);

    SpectralQuantity
    operator*(f32 val) const;

    SpectralQuantity &
    operator*=(f32 val);

    SpectralQuantity
    operator/(f32 div) const;

    SpectralQuantity &
    operator/=(f32 div);

    SpectralQuantity
    operator/(const SpectralQuantity &other) const;

    const f32 &
    operator[](u32 index) const;

    f32 &
    operator[](u32 index);

private:
    std::array<f32, N_SPECTRUM_SAMPLES> vals{};
};

using spectral = SpectralQuantity;

#endif // PT_SPECTRAL_QUANTITY_H
