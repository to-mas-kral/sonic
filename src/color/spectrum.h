#ifndef PT_SPECTRUM_H
#define PT_SPECTRUM_H

#include "../utils/chunk_allocator.h"
#include "cie_spectrums.h"
#include "color_space.h"
#include "sampled_spectrum.h"
#include "spectrum_consts.h"

class DenseSpectrum {
public:
    constexpr static DenseSpectrum
    make(const std::array<f32, LAMBDA_RANGE> &data) {
        DenseSpectrum ds{};
        ds.vals = data.data();

        return ds;
    }

    f32
    eval_single(f32 lambda) const;

    SampledSpectrum
    eval(const SampledLambdas &sl) const;

private:
    const f32 *vals;
};

constexpr DenseSpectrum CIE_X = DenseSpectrum::make(CIE_X_RAW);
constexpr DenseSpectrum CIE_Y = DenseSpectrum::make(CIE_Y_RAW);
constexpr DenseSpectrum CIE_Z = DenseSpectrum::make(CIE_Z_RAW);
constexpr DenseSpectrum CIE_65 = DenseSpectrum::make(CIE_D65_RAW);

class PiecewiseSpectrum {
public:
    static constexpr PiecewiseSpectrum
    make(const std::span<const f32> &data) {
        PiecewiseSpectrum ds{};

        if (data.size() % 2 != 0 || data.size() < 2) {
            throw std::runtime_error("Piecewise spectrum data is wrong");
        }

        ds.vals = data.data();
        ds.size = data.size();

        return ds;
    }

    f32
    eval_single(f32 lambda) const;

    inline SampledSpectrum
    eval(const SampledLambdas &sl) const;

private:
    const f32 *vals;
    u32 size{0};
};

class ConstantSpectrum {
public:
    static constexpr ConstantSpectrum
    make(f32 val) {
        ConstantSpectrum cs{};
        cs.val = val;
        return cs;
    }

    f32
    eval_single(f32 lambda) const;

    inline SampledSpectrum
    eval(const SampledLambdas &sl) const;

private:
    f32 val;
};

/// Bounded reflectance spectrum [0; 1]
struct RgbSpectrum {
    static RgbSpectrum
    make(const tuple3 &rgb);

    static RgbSpectrum
    from_coeff(const tuple3 &sigmoig_coeff);

    static RgbSpectrum
    make_empty();

    f32
    eval_single(f32 lambda) const;

    spectral
    eval(const SampledLambdas &lambdas) const;

    tuple3 sigmoid_coeff = tuple3(0.f);
};

struct RgbSpectrumUnbounded : RgbSpectrum {
    static RgbSpectrumUnbounded
    make(const tuple3 &rgb);

    f32
    eval_single(f32 lambda) const;

    spectral
    eval(const SampledLambdas &lambdas) const;

    f32 scale = 1.f;
};

struct RgbSpectrumIlluminant : RgbSpectrumUnbounded {
    static RgbSpectrumIlluminant
    make(const tuple3 &rgb, ColorSpace color_space);

    f32
    eval_single(f32 lambda) const;

    spectral
    eval(const SampledLambdas &lambdas) const;

    ColorSpace color_space = ColorSpace::sRGB;
};

struct BlackbodySpectrum {
    static BlackbodySpectrum
    make(i32 temp);

    f32
    eval_single(f32 lambda) const;

    spectral
    eval(const SampledLambdas &lambdas) const;

    i32 temp{0};
};

enum class SpectrumType {
    Constant,
    Dense,
    PiecewiseLinear,
    Rgb,
    RgbUnbounded,
    RgbIlluminant,
    Blackbody,
};

struct Spectrum {
    explicit
    Spectrum(DenseSpectrum ds)
        : type{SpectrumType::Dense}, dense_spectrum{ds} {}

    explicit
    Spectrum(PiecewiseSpectrum ps)
        : type{SpectrumType::PiecewiseLinear} {
        piecewise_spectrum = ps;
    }

    explicit
    Spectrum(ConstantSpectrum cs)
        : type{SpectrumType::Constant}, constant_spectrum{cs} {}

    explicit
    Spectrum(RgbSpectrum rs)
        : type{SpectrumType::Rgb} {
        rgb_spectrum = rs;
    }

    explicit
    Spectrum(RgbSpectrumUnbounded rs)
        : type{SpectrumType::RgbUnbounded} {
        rgb_spectrum_unbounded = rs;
    }

    explicit
    Spectrum(RgbSpectrumIlluminant rs)
        : type{SpectrumType::RgbIlluminant} {
        rgb_spectrum_illuminant = rs;
    }

    explicit
    Spectrum(BlackbodySpectrum bs)
        : type{SpectrumType::Blackbody} {
        blackbody_spectrum = bs;
    }

    SampledSpectrum
    eval(const SampledLambdas &lambdas) const;

    f32
    eval_single(f32 lambda) const;

    f32
    power() const;

    SpectrumType type;
    union {
        DenseSpectrum dense_spectrum;
        PiecewiseSpectrum piecewise_spectrum;
        ConstantSpectrum constant_spectrum{};
        RgbSpectrum rgb_spectrum;
        RgbSpectrumUnbounded rgb_spectrum_unbounded;
        RgbSpectrumIlluminant rgb_spectrum_illuminant;
        BlackbodySpectrum blackbody_spectrum;
    };
};

#endif // PT_SPECTRUM_H
