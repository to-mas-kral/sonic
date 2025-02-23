
#include "rgb2spec.h"

#include "../utils/algs.h"
#include "spectrum.h"

#include "../utils/panic.h"

#include <cassert>

// TODO: FIXME - can't catch exceptions in static init
static const auto rgb2spec = RGB2Spec("rgb2spec.out");

RgbSpectrum::RgbSpectrum(const SigmoigCoeff &sigmoig_coeff)
    : sigmoid_coeff(sigmoig_coeff) {}

RgbSpectrum
RgbSpectrum::from_rgb(const tuple3 &rgb) {
    assert(rgb.max_component() <= 1.F);
    assert(rgb.min_component() >= 0.F);
    return RgbSpectrum(rgb2spec.fetch(rgb));
}

RgbSpectrum
RgbSpectrum::from_coeff(const SigmoigCoeff &sigmoig_coeff) {
    return RgbSpectrum(sigmoig_coeff);
}

f32
RgbSpectrum::eval_single(const f32 lambda) const {
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

namespace {

tuple3
adjust_rgb(const tuple3 &rgb) {
    auto rgb_copy = rgb;
    const f32 scale = 2.F * rgb_copy.max_component();

    if (scale != 0.F) {
        rgb_copy /= scale;
    } else {
        rgb_copy = tuple3(0.F);
    }

    return rgb_copy;
}
} // namespace

RgbSpectrumUnbounded::RgbSpectrumUnbounded(const tuple3 &rgb)
    : RgbSpectrum(rgb2spec.fetch(adjust_rgb(rgb))), scale(2.F * rgb.max_component()) {
    assert(rgb.min_component() >= 0.f);
}

f32
RgbSpectrumUnbounded::eval_single(const f32 lambda) const {
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

RgbSpectrumIlluminant::RgbSpectrumIlluminant(const tuple3 &rgb,
                                             const ColorSpace color_space)
    : RgbSpectrumUnbounded(rgb), color_space(color_space) {}

f32
RgbSpectrumIlluminant::eval_single(const f32 lambda) const {
    f32 res = scale * RGB2Spec::eval(sigmoid_coeff, lambda);
    const DenseSpectrum *illuminant = nullptr;
    switch (color_space) {
    case ColorSpace::sRGB:
        illuminant = &CIE_65;
        break;
    default:
        panic();
    }

    /// TODO: this is a hack for normalizing the D65 illuminant to having a luminance
    /// of 1
    res *= illuminant->eval_single(lambda) * (CIE_Y_INTEGRAL / 10789.7637F);

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

BlackbodySpectrum::BlackbodySpectrum(const i32 temp) : temp(temp) {
    assert(temp > 0);
    assert(temp <= 12000);
}

f32
BlackbodySpectrum::eval_single(const f32 lambda) const {
    // taken from PBRT-v4
    const auto blackbody = [this](const f32 inner_lambda) {
        constexpr auto kb = 1.3806488e-23F;

        constexpr auto h = 6.62606957e-34F;
        constexpr auto c = 299792458.F;

        const auto l = inner_lambda * 1e-9F;
        const auto lambda5 = sqr(l) * sqr(l) * l;

        return 2.F * h * sqr(c) / (lambda5 * std::expm1(h * c / (l * kb * temp)));
    };

    const auto le = blackbody(lambda);

    const auto lambda_max = 2.8977721e-3F / temp * 1e9F;
    const auto max_l = blackbody(lambda_max);

    return le / max_l;
}

spectral
BlackbodySpectrum::eval(const SampledLambdas &lambdas) const {
    SpectralQuantity sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq[i] = eval_single(lambdas[i]);
    }

    return sq;
}

f32
DenseSpectrum::eval_single(const f32 lambda) const {
    assert(lambda >= LAMBDA_MIN && lambda <= LAMBDA_MAX);
    const u32 index = lround(lambda) - LAMBDA_MIN;
    return vals[index];
}

SpectralQuantity
DenseSpectrum::eval(const SampledLambdas &sl) const {
    SpectralQuantity sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq[i] = eval_single(sl[i]);
    }

    return sq;
}

f32
PiecewiseSpectrum::eval_single(const f32 lambda) const {
    assert(lambda >= LAMBDA_MIN && lambda <= LAMBDA_MAX);
    if (lambda < m_vals[0] || lambda > m_vals[m_vals.size() - 2]) {
        // Values outside of the range get mapped to 0
        return 0.F;
    }

    const auto index =
        2 * binary_search_interval(
                m_vals.size() / 2, [&](const size_t i) { return m_vals[i * 2]; }, lambda);

    assert(index + 3 < m_vals.size());
    const auto lambda_start = m_vals[index];
    const auto lambda_end = m_vals[index + 2];
    assert(lambda >= lambda_start && lambda <= lambda_end && lambda_end > lambda_start);

    const auto t = (lambda - lambda_start) / (lambda_end - lambda_start);
    const auto val_start = m_vals[index + 1];
    const auto val_end = m_vals[index + 3];

    return lerp(t, val_start, val_end);
}

SpectralQuantity
PiecewiseSpectrum::eval(const SampledLambdas &sl) const {
    SpectralQuantity sq{};
    for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
        sq[i] = eval_single(sl[i]);
    }

    return sq;
}

f32
ConstantSpectrum::eval_single() const {
    return val;
}

SpectralQuantity
ConstantSpectrum::eval() const {
    return SpectralQuantity::make_constant(val);
}

SpectralQuantity
Spectrum::eval(const SampledLambdas &lambdas) const {
    switch (type) {
    case SpectrumType::Constant:
        return constant_spectrum.eval();
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
        panic();
    }
}

f32
Spectrum::eval_single(const f32 lambda) const {
    switch (type) {
    case SpectrumType::Constant:
        return constant_spectrum.eval_single();
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
        panic();
    }
}

f32
Spectrum::power() const {
    // Just use the rectangle rule...
    f32 sum = 0.F;
    constexpr u32 num_steps = 100;
    constexpr f32 lambda_min = static_cast<f32>(LAMBDA_MIN);
    constexpr f32 lambda_max = static_cast<f32>(LAMBDA_MAX);
    constexpr f32 h = (lambda_max - lambda_min) / static_cast<f32>(num_steps);
    for (u32 i = 0; i < num_steps; i++) {
        const f32 lambda = lambda_min + (static_cast<f32>(i) * h) + (h / 2.F);
        sum += eval_single(lambda);
    }

    const f32 integral = sum * h;
    return integral / (lambda_max - lambda_min);
}
