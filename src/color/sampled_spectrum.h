#ifndef PT_SAMPLED_SPECTRUM_H
#define PT_SAMPLED_SPECTRUM_H

#include "../math/vecmath.h"
#include "../utils/basic_types.h"

#include <array>
#include <string>

constexpr u32 N_SPECTRUM_SAMPLES = 4;

struct SampledSpectrum {
    SampledSpectrum() = default;

    explicit SampledSpectrum(const std::array<f32, N_SPECTRUM_SAMPLES> &p_vals);

    static SampledSpectrum
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
    div_pdf(f32 pdf);

    std::string
    to_str() const;

    void
    clamp(f32 low, f32 high);

    static SampledSpectrum
    ONE();

    static SampledSpectrum
    ZERO();

    SampledSpectrum
    operator+(const SampledSpectrum &other) const;

    SampledSpectrum &
    operator+=(const SampledSpectrum &other);

    SampledSpectrum
    operator-(const SampledSpectrum &other) const;

    SampledSpectrum &
    operator-=(const SampledSpectrum &other);

    SampledSpectrum
    operator*(const SampledSpectrum &other) const;

    SampledSpectrum &
    operator*=(const SampledSpectrum &other);

    SampledSpectrum
    operator*(f32 val) const;

    SampledSpectrum &
    operator*=(f32 val);

    SampledSpectrum
    operator/(f32 div) const;

    SampledSpectrum
    operator/(const SampledSpectrum &other) const;

    const f32&
    operator[](u32 index) const;

    f32&
    operator[](u32 index);

    std::array<f32, N_SPECTRUM_SAMPLES> vals{};
};

struct SampledLambdas {
    static SampledLambdas
    new_sample_uniform(f32 rand);

    static SampledLambdas
    new_mock();

    vec3
    to_xyz(const SampledSpectrum &radiance) const;

    void
    terminate_secondary();

    const f32 &
    operator[](u32 index) const;

    bool is_secondary_terminated{false};
    std::array<f32, N_SPECTRUM_SAMPLES> lambdas{};
};

using spectral = SampledSpectrum;

#endif // PT_SAMPLED_SPECTRUM_H
