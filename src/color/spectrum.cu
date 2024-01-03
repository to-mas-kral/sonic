
#include "rgb2spec.h"
#include "spectrum.h"

static const RGB2Spec rgb2spec = RGB2Spec("resources/rgb2spec.out");

RgbSpectrum
RgbSpectrum::make(const tuple3 &rgb) {
    RgbSpectrum spectrum{
        .sigmoid_coeff = rgb2spec.fetch(rgb),
    };

    return spectrum;
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

RgbSpectrumIlluminant
RgbSpectrumIlluminant::make(const tuple3 &_rgb, ColorSpace color_space) {
    auto spectrum_unbounded = RgbSpectrumUnbounded::make(_rgb);
    RgbSpectrumIlluminant spectrum{};
    spectrum.sigmoid_coeff = spectrum_unbounded.sigmoid_coeff;
    spectrum.scale = spectrum_unbounded.scale;
    spectrum.color_space = color_space;

    return spectrum;
}
