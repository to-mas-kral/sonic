#ifndef PT_SPECTRUM_H
#define PT_SPECTRUM_H

#include "color_space.h"
#include "rgb2spec.h"
#include "sampled_spectrum.h"

#include <cassert>

class DenseSpectrum {
public:
    constexpr static DenseSpectrum
    make(const Array<f32, LAMBDA_RANGE> &data) {
        DenseSpectrum ds{};
        ds.vals = data.data();

        return ds;
    }

    f32
    eval_single(f32 lambda) const {
        assert(lambda >= LAMBDA_MIN && lambda <= LAMBDA_MAX);
        u32 index = lround(lambda) - LAMBDA_MIN;
        return vals[index];
    }

    inline SampledSpectrum
    eval(const SampledLambdas &sl) const {
        SampledSpectrum sq{};
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            sq[i] = eval_single(sl.lambdas[i]);
        }

        return sq;
    }

private:
    const f32 *vals;
};

const DenseSpectrum CIE_X = DenseSpectrum::make(CIE_X_RAW);
const DenseSpectrum CIE_Y = DenseSpectrum::make(CIE_Y_RAW);
const DenseSpectrum CIE_Z = DenseSpectrum::make(CIE_Z_RAW);
const DenseSpectrum CIE_65 = DenseSpectrum::make(CIE_D65_RAW);

inline vec3
SampledLambdas::to_xyz(const SampledSpectrum &radiance) {
    SampledSpectrum x = CIE_X.eval(static_cast<const SampledLambdas &>(*this)) * radiance;
    SampledSpectrum y = CIE_Y.eval(static_cast<const SampledLambdas &>(*this)) * radiance;
    SampledSpectrum z = CIE_Z.eval(static_cast<const SampledLambdas &>(*this)) * radiance;

    x.div_pdf(PDF);
    y.div_pdf(PDF);
    z.div_pdf(PDF);

    f32 x_xyz = x.average() / CIE_Y_INTEGRAL;
    f32 y_xyz = y.average() / CIE_Y_INTEGRAL;
    f32 z_xyz = z.average() / CIE_Y_INTEGRAL;
    return vec3(x_xyz, y_xyz, z_xyz);
}

class PiecewiseSpectrum {
public:
    static PiecewiseSpectrum
    make(const Span<f32> &data) {
        PiecewiseSpectrum ds{};

        if (data.size() % 2 != 0 || data.size() < 2) {
            throw std::runtime_error("Piecewise spectrum data is wrong");
        }

        ds.vals = data.data();
        ds.size_half = data.size() / 2;

        return ds;
    }

    f32
    eval_single(f32 lambda) const {
        assert(lambda >= LAMBDA_MIN && lambda <= LAMBDA_MAX);
        u32 index =
            binary_search_interval(size_half, [&](size_t i) { return vals[i]; }, lambda);
        return vals[(size_half) + index];
    }

    inline SampledSpectrum
    eval(const SampledLambdas &sl) const {
        SampledSpectrum sq{};
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            sq[i] = eval_single(sl.lambdas[i]);
        }

        return sq;
    }

private:
    const f32 *vals;
    u32 size_half;
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
    eval_single(f32 lambda) const {
        return val;
    }

    inline SampledSpectrum
    eval(const SampledLambdas &sl) const {
        return SampledSpectrum::make_constant(val);
    }

private:
    f32 val;
};

/// Bounded reflectance spectrum [0; 1]
struct RgbSpectrum {
    static RgbSpectrum
    make(const tuple3 &rgb);

    static RgbSpectrum
    from_coeff(const tuple3 &sigmoig_coeff) {
        return RgbSpectrum{
            .sigmoid_coeff = sigmoig_coeff,
        };
    }

    static RgbSpectrum
    make_empty() {
        return RgbSpectrum{
            .sigmoid_coeff = tuple3(0.f),
        };
    }

    f32
    eval_single(f32 lambda) const {
        return RGB2Spec::eval(sigmoid_coeff, lambda);
    }

    spectral
    eval(const SampledLambdas &lambdas) const {
        spectral sq{};
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            sq[i] = eval_single(lambdas[i]);
        }

        return sq;
    }

    tuple3 sigmoid_coeff = tuple3(0.f);
};

struct RgbSpectrumUnbounded : public RgbSpectrum {
    static RgbSpectrumUnbounded
    make(const tuple3 &rgb);

    f32
    eval_single(f32 lambda) const {
        return scale * RGB2Spec::eval(sigmoid_coeff, lambda);
    }

    spectral
    eval(const SampledLambdas &lambdas) const {
        spectral sq{};
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            sq[i] = eval_single(lambdas[i]);
        }

        return sq;
    }

    f32 scale = 1.f;
};

struct RgbSpectrumIlluminant : public RgbSpectrumUnbounded {
    static RgbSpectrumIlluminant
    make(const tuple3 &rgb, ColorSpace color_space);

    f32
    eval_single(f32 lambda) const {
        f32 res = scale * RGB2Spec::eval(sigmoid_coeff, lambda);
        const DenseSpectrum *illuminant = nullptr;
        switch (color_space) {
        case ColorSpace::sRGB:
            illuminant = &CIE_65;
            break;
        default:
            assert(false);
        }

        /// TODO: this is a hack for normalizing the D65 illuminant to having a luminance
        /// of 1
        res *= illuminant->eval_single(lambda) * (CIE_Y_INTEGRAL / 10789.7637f);

        return res;
    }

    spectral
    eval(const SampledLambdas &lambdas) const {
        spectral sq{};
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            sq[i] = eval_single(lambdas[i]);
        }

        return sq;
    }

    // TODO: best to store this out-of-band in the case if illuminant textures...
    ColorSpace color_space = ColorSpace::sRGB;
};

enum class SpectrumType {
    Constant,
    Dense,
    PiecewiseLinear,
    Rgb,
    RgbUnbounded,
};

struct Spectrum {
    explicit Spectrum(DenseSpectrum ds) : type{SpectrumType::Dense}, dense_spectrum{ds} {}

    explicit Spectrum(PiecewiseSpectrum ps)
        : type{SpectrumType::PiecewiseLinear}, piecewise_spectrum{ps} {}

    explicit Spectrum(ConstantSpectrum cs)
        : type{SpectrumType::Constant}, constant_spectrum{cs} {}

    explicit Spectrum(RgbSpectrum rs) : type{SpectrumType::Rgb}, rgb_spectrum{rs} {}

    explicit Spectrum(RgbSpectrumUnbounded rs)
        : type{SpectrumType::RgbUnbounded}, rgb_spectrum_unbounded{rs} {}

    SampledSpectrum
    eval(const SampledLambdas &lambdas) const {
        switch (type) {
        case SpectrumType::Constant:
            return constant_spectrum.eval(lambdas);
        case SpectrumType::Dense:
            return dense_spectrum.eval(lambdas);
        case SpectrumType::PiecewiseLinear:
            return piecewise_spectrum.eval(lambdas);
        case SpectrumType::Rgb:
            return rgb_spectrum.eval(lambdas);
        case SpectrumType::RgbUnbounded:
            return rgb_spectrum_unbounded.eval(lambdas);
        default:
            assert(false);
        }
    }

    f32
    eval_single(f32 lambda) const {
        switch (type) {
        case SpectrumType::Constant:
            return constant_spectrum.eval_single(lambda);
        case SpectrumType::Dense:
            return dense_spectrum.eval_single(lambda);
        case SpectrumType::PiecewiseLinear:
            return piecewise_spectrum.eval_single(lambda);
        case SpectrumType::Rgb:
            return rgb_spectrum.eval_single(lambda);
        case SpectrumType::RgbUnbounded:
            return rgb_spectrum_unbounded.eval_single(lambda);
        default:
            assert(false);
        }
    }

    SpectrumType type;
    union {
        DenseSpectrum dense_spectrum;
        PiecewiseSpectrum piecewise_spectrum;
        ConstantSpectrum constant_spectrum{};
        RgbSpectrum rgb_spectrum;
        RgbSpectrumUnbounded rgb_spectrum_unbounded;
    };
};

#endif // PT_SPECTRUM_H
