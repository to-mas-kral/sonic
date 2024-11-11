#ifndef PT_SPECTRUM_H
#define PT_SPECTRUM_H

#include "cie_spectrums.h"
#include "color_space.h"
#include "sampled_spectrum.h"
#include "spectrum_consts.h"

class DenseSpectrum {
public:
    explicit constexpr
    DenseSpectrum(const std::array<f32, LAMBDA_RANGE> &vals)
        : vals(vals.data()) {}

    f32
    eval_single(f32 lambda) const;

    SampledSpectrum
    eval(const SampledLambdas &sl) const;

private:
    const f32 *vals;
};

constexpr auto CIE_X = DenseSpectrum(CIE_X_RAW);
constexpr auto CIE_Y = DenseSpectrum(CIE_Y_RAW);
constexpr auto CIE_Z = DenseSpectrum(CIE_Z_RAW);
constexpr auto CIE_65 = DenseSpectrum(CIE_D65_RAW);

class PiecewiseSpectrum {
public:
    explicit constexpr
    PiecewiseSpectrum(const std::span<const f32> &vals)
        : vals{vals.data()}, size{static_cast<u32>(vals.size())} {
        assert(vals.size() < std::numeric_limits<u32>::max());
        if (vals.size() % 2 != 0 || vals.size() < 2) {
            throw std::runtime_error("Piecewise spectrum data is wrong");
        }
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
    explicit constexpr
    ConstantSpectrum(const f32 val)
        : val(val) {}

    f32
    eval_single() const;

    inline SampledSpectrum
    eval() const;

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

    tuple3 sigmoid_coeff = tuple3(0.F);
};

struct RgbSpectrumUnbounded : RgbSpectrum {
    static RgbSpectrumUnbounded
    make(tuple3 rgb);

    f32
    eval_single(f32 lambda) const;

    spectral
    eval(const SampledLambdas &lambdas) const;

    f32 scale = 1.F;
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

enum class SpectrumType : u8 {
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
        : type{SpectrumType::PiecewiseLinear}, piecewise_spectrum{ps} {}

    explicit
    Spectrum(ConstantSpectrum cs)
        : type{SpectrumType::Constant}, constant_spectrum{cs} {}

    explicit
    Spectrum(RgbSpectrum rs)
        : type{SpectrumType::Rgb}, rgb_spectrum{rs} {}

    explicit
    Spectrum(RgbSpectrumUnbounded rs)
        : type{SpectrumType::RgbUnbounded}, rgb_spectrum_unbounded(rs) {}

    explicit
    Spectrum(RgbSpectrumIlluminant rs)
        : type{SpectrumType::RgbIlluminant}, rgb_spectrum_illuminant(rs) {}

    explicit
    Spectrum(BlackbodySpectrum bs)
        : type{SpectrumType::Blackbody}, blackbody_spectrum{bs} {}

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
        ConstantSpectrum constant_spectrum;
        RgbSpectrum rgb_spectrum;
        RgbSpectrumUnbounded rgb_spectrum_unbounded;
        RgbSpectrumIlluminant rgb_spectrum_illuminant;
        BlackbodySpectrum blackbody_spectrum;
    };
};

#endif // PT_SPECTRUM_H
