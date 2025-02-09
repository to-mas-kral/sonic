#ifndef PT_SPECTRUM_H
#define PT_SPECTRUM_H

#include "cie_spectra.h"
#include "color_space.h"
#include "rgb2spec.h"
#include "sampled_lambdas.h"
#include "spectral_quantity.h"
#include "spectrum_consts.h"

class DenseSpectrum {
public:
    explicit constexpr
    DenseSpectrum(const std::array<f32, LAMBDA_RANGE> &vals)
        : vals(vals.data()) {}

    f32
    eval_single(f32 lambda) const;

    SpectralQuantity
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

    inline SpectralQuantity
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

    inline SpectralQuantity
    eval() const;

private:
    f32 val;
};

/// Bounded reflectance spectrum [0; 1]
class RgbSpectrum {
public:
    static RgbSpectrum
    from_rgb(const tuple3 &rgb);

    static RgbSpectrum
    from_coeff(const SigmoigCoeff &sigmoig_coeff);

    f32
    eval_single(f32 lambda) const;

    spectral
    eval(const SampledLambdas &lambdas) const;

protected:
    explicit
    RgbSpectrum(const SigmoigCoeff &sigmoig_coeff);

    SigmoigCoeff sigmoid_coeff;
};

class RgbSpectrumUnbounded : protected RgbSpectrum {
public:
    explicit
    RgbSpectrumUnbounded(const tuple3 &rgb);

    f32
    eval_single(f32 lambda) const;

    spectral
    eval(const SampledLambdas &lambdas) const;

protected:
    f32 scale = 1.F;
};

class RgbSpectrumIlluminant : protected RgbSpectrumUnbounded {
public:
    explicit
    RgbSpectrumIlluminant(const tuple3 &rgb, ColorSpace color_space = ColorSpace::sRGB);

    f32
    eval_single(f32 lambda) const;

    spectral
    eval(const SampledLambdas &lambdas) const;

protected:
    ColorSpace color_space = ColorSpace::sRGB;
};

class BlackbodySpectrum {
public:
    explicit
    BlackbodySpectrum(i32 temp);

    f32
    eval_single(f32 lambda) const;

    spectral
    eval(const SampledLambdas &lambdas) const;

protected:
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

class Spectrum {
public:
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

    SpectralQuantity
    eval(const SampledLambdas &lambdas) const;

    f32
    eval_single(f32 lambda) const;

    f32
    power() const;

private:
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
