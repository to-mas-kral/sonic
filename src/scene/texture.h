#ifndef PT_TEXTURE_H
#define PT_TEXTURE_H

#include "image.h"

#include "../color/spectrum.h"
#include "../math/vecmath.h"
#include "../utils/basic_types.h"
#include "../utils/panic.h"

// PBRTv4 texture params
struct ImageTextureParams {
    f32 scale{1.f};
    f32 uscale{1.f};
    f32 vscale{1.f};
    f32 udelta{0.f};
    f32 vdelta{0.f};
    bool invert{false};
};

enum class TextureSpectrumType : u8 {
    Rgb,
    Unbounded,
    Illuminant,
};

/// UV - texture space:
///
///  1   ^
///      |
///      |
///      |
///      |
///      |
///      |
///      |
///    --.--------------->
///  0,0 |               1
class ImageTexture {
public:
    explicit
    ImageTexture(Image *image, const TextureSpectrumType spectrum_type,
                 const ImageTextureParams &params = ImageTextureParams())
        : spectrum_type{spectrum_type}, image{image}, params{params} {}

    explicit
    ImageTexture(Image *image, const ImageTextureParams &params = ImageTextureParams())
        : image{image}, params{params} {}

    tuple3
    fetch_rgb_texel(const uvec2 &coords) const {
        return image->fetch_rgb_texel(coords);
    }

    /// These coords are top to bottom!:
    ///
    /// 0,0 |               x
    ///   --.--------------->
    ///     |
    ///     |
    ///     |
    ///     |
    ///     |
    ///     |
    ///  y  |
    tuple3
    fetch_rgb_texel_xycoords(const vec2 &coords) const {
        return image->fetch_rgb_texel(coords);
    }

    tuple3
    fetch_rgb(const vec2 &uv) const {
        // Remap UV because image coordinate system is flipped.
        // It's better to do this here because if flipping is done
        // in the image, there are inaccuracies that affect envmap
        // pdf sampling calculations...
        const auto remapped_uv = vec2(uv.x, 1.f - uv.y);
        return image->fetch_rgb(remapped_uv, params);
    }

    f32
    fetch_float(const vec2 &uv) const {
        // See fetch_rgb comment.
        const auto remapped_uv = vec2(uv.x, 1.f - uv.y);
        return image->fetch_float(remapped_uv, params);
    }

    const ColorSpace &
    color_space() const {
        return image->get_scolor_space();
    }

    i32
    width() const {
        return image->get_width();
    }

    i32
    height() const {
        return image->get_height();
    }

    SampledSpectrum
    fetch_spectrum(const vec2 &uv, const SampledLambdas &lambdas) const {
        const auto rgb = fetch_rgb(uv);

        switch (spectrum_type) {
        case TextureSpectrumType::Rgb:
            return RgbSpectrum::make(rgb).eval(lambdas);
        case TextureSpectrumType::Unbounded:
            return RgbSpectrumUnbounded::make(rgb).eval(lambdas);
        case TextureSpectrumType::Illuminant:
            return RgbSpectrumIlluminant::make(rgb, color_space()).eval(lambdas);
        default:
            panic();
        }
    }

private:
    // Only used if the image holds spectra
    TextureSpectrumType spectrum_type{};
    Image *image;
    ImageTextureParams params;
};

enum class TextureType : u8 {
    Constant,
    Image,
    Scale,
    Mix,
};

class FloatTexture;

class FloatScaleTexture {
public:
    FloatScaleTexture(FloatTexture *texture, FloatTexture *scale_tex)
        : texture{texture}, scale_tex{scale_tex} {}

    f32
    fetch(const vec2 &uv) const;

private:
    FloatTexture *texture{nullptr};
    FloatTexture *scale_tex{nullptr};
};

class FloatMixTexture {
public:
    FloatMixTexture(FloatTexture *tex1, FloatTexture *tex2, const f32 mix)
        : mix{mix}, tex1{tex1}, tex2{tex2} {}

    f32
    fetch(const vec2 &uv) const;

private:
    f32 mix{0.5f};
    FloatTexture *tex1{nullptr};
    FloatTexture *tex2{nullptr};
};

class FloatTexture {
public:
    FloatTexture() = default;

    static FloatTexture
    make(const ImageTexture &image_texture) {
        FloatTexture tex{};
        tex.texture_type = TextureType::Image;
        tex.inner.image = image_texture;

        return tex;
    }

    static FloatTexture
    make(const FloatScaleTexture texture) {
        FloatTexture tex{};
        tex.texture_type = TextureType::Scale;
        tex.inner.scale_texture = texture;

        return tex;
    }

    static FloatTexture
    make(const FloatMixTexture &p_tex) {
        FloatTexture tex{};
        tex.texture_type = TextureType::Mix;
        tex.inner.mix_texture = p_tex;

        return tex;
    }

    static FloatTexture
    make(const f32 value) {
        FloatTexture tex{};
        tex.texture_type = TextureType::Constant;
        tex.inner.constant_f32 = value;

        return tex;
    }

    f32
    fetch(const vec2 &uv) const {
        switch (texture_type) {
        case TextureType::Constant:
            return inner.constant_f32;
        case TextureType::Image:
            return inner.image.fetch_float(uv);
        case TextureType::Scale:
            return inner.scale_texture.fetch(uv);
        case TextureType::Mix:
            return inner.mix_texture.fetch(uv);
        default:
            panic();
        }
    }

    TextureType texture_type{};
    union {
        f32 constant_f32;
        ImageTexture image;
        FloatScaleTexture scale_texture;
        FloatMixTexture mix_texture;
    } inner{};
};

class SpectrumTexture;

class SpectrumScaleTexture {
public:
    SpectrumScaleTexture(SpectrumTexture *texture, FloatTexture *scale_tex)
        : texture{texture}, scale_tex{scale_tex} {}

    SampledSpectrum
    fetch(const vec2 &uv, const SampledLambdas &lambdas) const;

private:
    SpectrumTexture *texture{nullptr};
    FloatTexture *scale_tex{nullptr};
};

class SpectrumMixTexture {
public:
    SpectrumMixTexture(SpectrumTexture *tex1, SpectrumTexture *tex2, const f32 mix)
        : mix{mix}, tex1{tex1}, tex2{tex2} {}

    SampledSpectrum
    fetch(const vec2 &uv, const SampledLambdas &lambdas) const;

private:
    f32 mix{0.5f};
    SpectrumTexture *tex1{nullptr};
    SpectrumTexture *tex2{nullptr};
};

class SpectrumTexture {
public:
    SpectrumTexture() = default;

    static SpectrumTexture
    make(const ImageTexture &image_texture) {
        SpectrumTexture tex{};
        tex.texture_type = TextureType::Image;
        tex.inner.image = image_texture;

        return tex;
    }

    static SpectrumTexture
    make(const SpectrumScaleTexture texture) {
        SpectrumTexture tex{};
        tex.texture_type = TextureType::Scale;
        tex.inner.scale_texture = texture;

        return tex;
    }

    static SpectrumTexture
    make(const SpectrumMixTexture &p_tex) {
        SpectrumTexture tex{};
        tex.texture_type = TextureType::Mix;
        tex.inner.mix_texture = p_tex;

        return tex;
    }

    static SpectrumTexture
    make(const RgbSpectrum value) {
        SpectrumTexture tex{};
        tex.texture_type = TextureType::Constant;
        tex.inner.constant_spectrum = Spectrum(value);

        return tex;
    }

    static SpectrumTexture
    make(const RgbSpectrumUnbounded value) {
        SpectrumTexture tex{};
        tex.texture_type = TextureType::Constant;
        tex.inner.constant_spectrum = Spectrum(value);

        return tex;
    }

    static SpectrumTexture
    make(const Spectrum &value) {
        SpectrumTexture tex{};
        tex.texture_type = TextureType::Constant;
        tex.inner.constant_spectrum = value;

        return tex;
    }

    SampledSpectrum
    fetch(const vec2 &uv, const SampledLambdas &lambdas) const {
        switch (texture_type) {
        case TextureType::Constant:
            return inner.constant_spectrum.eval(lambdas);
        case TextureType::Image: {
            return inner.image.fetch_spectrum(uv, lambdas);
        }
        case TextureType::Scale: {
            return inner.scale_texture.fetch(uv, lambdas);
        }
        case TextureType::Mix:
            return inner.mix_texture.fetch(uv, lambdas);
        default:
            panic();
        }
    }

    TextureType texture_type{};
    union {
        Spectrum constant_spectrum{ConstantSpectrum::make(0.f)};
        ImageTexture image;
        SpectrumScaleTexture scale_texture;
        SpectrumMixTexture mix_texture;
    } inner{};
};

#endif // PT_TEXTURE_H
