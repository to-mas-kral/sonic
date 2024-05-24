
#include "texture.h"

f32
FloatScaleTexture::fetch(const vec2 &uv) const {
    const auto scale = scale_tex->fetch(uv);
    return texture->fetch(uv) * scale;
}

f32
FloatMixTexture::fetch(const vec2 &uv) const {
    const auto t1 = tex1->fetch(uv);
    const auto t2 = tex2->fetch(uv);

    return lerp(mix, t1, t2);
}

SampledSpectrum
SpectrumScaleTexture::fetch(const vec2 &uv, const SampledLambdas &lambdas) const {
    const auto scale = scale_tex->fetch(uv);
    return texture->fetch(uv, lambdas) * scale;
}

SampledSpectrum
SpectrumMixTexture::fetch(const vec2 &uv, const SampledLambdas &lambdas) const {
    const auto t1 = tex1->fetch(uv, lambdas);
    const auto t2 = tex2->fetch(uv, lambdas);

    return lerp(mix, t1, t2);
}
