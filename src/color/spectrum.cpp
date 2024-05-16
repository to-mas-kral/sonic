
#include "rgb2spec.h"

#include "../utils/algs.h"
#include "spectrum.h"
#include <cassert>

static const RGB2Spec rgb2spec = RGB2Spec("rgb2spec.out");

RgbSpectrum
RgbSpectrum::make(const tuple3 &rgb) {
    RgbSpectrum spectrum{
        .sigmoid_coeff = rgb2spec.fetch(rgb),
    };

    return spectrum;
}

RgbSpectrum
RgbSpectrum::from_coeff(const tuple3 &sigmoig_coeff) {
    return RgbSpectrum{
        .sigmoid_coeff = sigmoig_coeff,
    };
}

RgbSpectrum
RgbSpectrum::make_empty() {
    return RgbSpectrum{
        .sigmoid_coeff = tuple3(0.f),
    };
}

f32
RgbSpectrum::eval_single(f32 lambda) const {
    return RGB2Spec::eval(sigmoid_coeff, lambda);
}

spectral
RgbSpectrum::eval(const SampledLambdas &lambdas) const {
    spectral sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq[i] = eval_single(lambdas[i]);
    }

    return sq;
}

RgbSpectrumUnbounded
RgbSpectrumUnbounded::make(const tuple3 &_rgb) {
    f32 scale = 2.f * _rgb.max_component();

    tuple3 rgb = _rgb;
    if (scale != 0.f) {
        rgb /= scale;
    } else {
        rgb = tuple3(0.f);
    }

    RgbSpectrumUnbounded spectrum{};
    spectrum.sigmoid_coeff = rgb2spec.fetch(rgb);
    spectrum.scale = scale;

    return spectrum;
}

f32
RgbSpectrumUnbounded::eval_single(f32 lambda) const {
    return scale * RGB2Spec::eval(sigmoid_coeff, lambda);
}

spectral
RgbSpectrumUnbounded::eval(const SampledLambdas &lambdas) const {
    spectral sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq[i] = eval_single(lambdas[i]);
    }

    return sq;
}

RgbSpectrumIlluminant
RgbSpectrumIlluminant::make(const tuple3 &_rgb, ColorSpace color_space) {
    auto spectrum_unbounded = RgbSpectrumUnbounded::make(_rgb);
    RgbSpectrumIlluminant spectrum{};
    spectrum.sigmoid_coeff = spectrum_unbounded.sigmoid_coeff;
    spectrum.scale = spectrum_unbounded.scale;
    spectrum.color_space = color_space;

    return spectrum;
}

f32
RgbSpectrumIlluminant::eval_single(f32 lambda) const {
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
RgbSpectrumIlluminant::eval(const SampledLambdas &lambdas) const {
    spectral sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq[i] = eval_single(lambdas[i]);
    }

    return sq;
}

BlackbodySpectrum
BlackbodySpectrum::make(const i32 temp) {
    return BlackbodySpectrum{.temp = temp};
}

f32
BlackbodySpectrum::eval_single(const f32 lambda) const {
    // taken from PBRT-v4
    const auto blackbody = [this](const f32 lambda) {
        constexpr auto kb = 1.3806488e-23f;

        constexpr auto h = 6.62606957e-34f;
        constexpr auto c = 299792458.f;

        const auto l = lambda * 1e-9f;
        const auto lambda5 = sqr(l) * sqr(l) * l;

        return 2.f * h * sqr(c) / (lambda5 * std::expm1(h * c / (l * kb * temp)));
    };

    const auto le = blackbody(lambda);

    const auto lambda_max = 2.8977721e-3f / temp * 1e9f;
    const auto max_l = blackbody(lambda_max);

    return le / max_l;
}

spectral
BlackbodySpectrum::eval(const SampledLambdas &lambdas) const {
    SampledSpectrum sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq[i] = eval_single(lambdas.lambdas[i]);
    }

    return sq;
}

f32
DenseSpectrum::eval_single(f32 lambda) const {
    assert(lambda >= LAMBDA_MIN && lambda <= LAMBDA_MAX);
    u32 index = lround(lambda) - LAMBDA_MIN;
    return vals[index];
}

SampledSpectrum
DenseSpectrum::eval(const SampledLambdas &sl) const {
    SampledSpectrum sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq[i] = eval_single(sl.lambdas[i]);
    }

    return sq;
}

PiecewiseSpectrum
PiecewiseSpectrum::make(const Span<f32> &data) {
    PiecewiseSpectrum ds{};

    if (data.size() % 2 != 0 || data.size() < 2) {
        throw std::runtime_error("Piecewise spectrum data is wrong");
    }

    ds.vals = data.data();
    ds.size = data.size();

    return ds;
}

f32
PiecewiseSpectrum::eval_single(const f32 lambda) const {
    assert(lambda >= LAMBDA_MIN && lambda <= LAMBDA_MAX);
    const auto index =
        binary_search_interval(size, [&](const size_t i) { return vals[2 * i]; }, lambda);
    return vals[2 * index + 1];
}

SampledSpectrum
PiecewiseSpectrum::eval(const SampledLambdas &sl) const {
    SampledSpectrum sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq[i] = eval_single(sl.lambdas[i]);
    }

    return sq;
}

f32
ConstantSpectrum::eval_single(f32 lambda) const {
    return val;
}

SampledSpectrum
ConstantSpectrum::eval(const SampledLambdas &sl) const {
    return SampledSpectrum::make_constant(val);
}

SampledSpectrum
Spectrum::eval(const SampledLambdas &lambdas) const {
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
    case SpectrumType::RgbIlluminant:
        return rgb_spectrum_illuminant.eval(lambdas);
    case SpectrumType::Blackbody:
        return blackbody_spectrum.eval(lambdas);
    default:
        assert(false);
    }
}

f32
Spectrum::eval_single(f32 lambda) const {
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
    case SpectrumType::RgbIlluminant:
        return rgb_spectrum_illuminant.eval_single(lambda);
    case SpectrumType::Blackbody:
        return blackbody_spectrum.eval_single(lambda);
    default:
        assert(false);
    }
}
