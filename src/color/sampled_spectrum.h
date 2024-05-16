#ifndef PT_SAMPLED_SPECTRUM_H
#define PT_SAMPLED_SPECTRUM_H

#include "../math/vecmath.h"
#include "../utils/basic_types.h"

constexpr u32 N_SPECTRUM_SAMPLES = 4;

struct SampledSpectrum {
    SampledSpectrum() = default;

    explicit SampledSpectrum(const Array<f32, N_SPECTRUM_SAMPLES> &p_vals);

    static SampledSpectrum
    make_constant(f32 constant);

    f32
    average();

    f32
    max_component() const;

    bool
    isnan() const;

    bool
    isinf() const;

    bool
    is_negative() const;

    void
    div_pdf(f32 pdf);

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

    f32 &
    operator[](u32 index);

    Array<f32, N_SPECTRUM_SAMPLES> vals;
};

struct SampledLambdas {
    static SampledLambdas
    new_sample_uniform(f32 rand);

    static SampledLambdas
    new_mock();

    vec3
    to_xyz(const SampledSpectrum &radiance);

    const f32 &
    operator[](u32 index) const;

    Array<f32, N_SPECTRUM_SAMPLES> lambdas;
};

using spectral = SampledSpectrum;

#endif // PT_SAMPLED_SPECTRUM_H
