#ifndef PT_TEXTURE_H
#define PT_TEXTURE_H

#include "image.h"

#include "../color/spectrum.h"
#include "../math/vecmath.h"
#include "../utils/basic_types.h"
#include "../utils/panic.h"

// PBRTv4 texture params
struct ImageTextureParams {
    f32 scale{1.F};
    f32 uscale{1.F};
    f32 vscale{1.F};
    f32 udelta{0.F};
    f32 vdelta{0.F};
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
        const auto remapped_uv = vec2(uv.x, 1.F - uv.y);
        return image->fetch_rgb(remapped_uv, params);
    }

    f32
    fetch_float(const vec2 &uv) const {
        // See fetch_rgb comment.
        const auto remapped_uv = vec2(uv.x, 1.F - uv.y);
        return image->fetch_float(remapped_uv, params);
    }

    const ColorSpace &
    color_space() const {
        return image->get_scolor_space();
    }

    i32
    width() const {
        return image->width();
    }

    i32
    height() const {
        return image->height();
    }

    SpectralQuantity
    fetch_spectrum(const vec2 &uv, const SampledLambdas &lambdas) const {
        const auto rgb = fetch_rgb(uv);

        switch (spectrum_type) {
        case TextureSpectrumType::Rgb:
            return RgbSpectrum::from_rgb(rgb).eval(lambdas);
        case TextureSpectrumType::Unbounded:
            return RgbSpectrumUnbounded(rgb).eval(lambdas);
        case TextureSpectrumType::Illuminant:
            return RgbSpectrumIlluminant(rgb, color_space()).eval(lambdas);
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
    f32 mix{0.5F};
    FloatTexture *tex1{nullptr};
    FloatTexture *tex2{nullptr};
};

class FloatTexture {
public:
    explicit
    FloatTexture(const ImageTexture &image_texture)
        : texture_type{TextureType::Image}, image_texture{image_texture} {}

    explicit
    FloatTexture(const FloatScaleTexture scale_texture)
        : texture_type{TextureType::Scale}, scale_texture{scale_texture} {}

    explicit
    FloatTexture(const FloatMixTexture &mix_texture)
        : texture_type{TextureType::Mix}, mix_texture{mix_texture} {}

    explicit
    FloatTexture(const f32 value)
        : texture_type{TextureType::Constant}, constant_f32{value} {}

    f32
    fetch(const vec2 &uv) const {
        switch (texture_type) {
        case TextureType::Constant:
            return constant_f32;
        case TextureType::Image:
            return image_texture.fetch_float(uv);
        case TextureType::Scale:
            return scale_texture.fetch(uv);
        case TextureType::Mix:
            return mix_texture.fetch(uv);
        default:
            panic();
        }
    }

private:
    TextureType texture_type{};
    union {
        f32 constant_f32{0.F};
        ImageTexture image_texture;
        FloatScaleTexture scale_texture;
        FloatMixTexture mix_texture;
    };
};

class SpectrumTexture;

class SpectrumScaleTexture {
public:
    SpectrumScaleTexture(SpectrumTexture *texture, FloatTexture *scale_tex)
        : texture{texture}, scale_tex{scale_tex} {}

    SpectralQuantity
    fetch(const vec2 &uv, const SampledLambdas &lambdas) const;

private:
    SpectrumTexture *texture{nullptr};
    FloatTexture *scale_tex{nullptr};
};

class SpectrumMixTexture {
public:
    SpectrumMixTexture(SpectrumTexture *tex1, SpectrumTexture *tex2, const f32 mix)
        : mix{mix}, tex1{tex1}, tex2{tex2} {}

    SpectralQuantity
    fetch(const vec2 &uv, const SampledLambdas &lambdas) const;

private:
    f32 mix{0.5F};
    SpectrumTexture *tex1{nullptr};
    SpectrumTexture *tex2{nullptr};
};

class SpectrumTexture {
public:
    explicit
    SpectrumTexture(const ImageTexture &image_texture)
        : texture_type(TextureType::Image), image_texture{image_texture} {}

    explicit
    SpectrumTexture(const SpectrumScaleTexture texture)
        : texture_type(TextureType::Scale), scale_texture{texture} {}

    explicit
    SpectrumTexture(const SpectrumMixTexture &mix_texture)
        : texture_type(TextureType::Mix), mix_texture{mix_texture} {}

    explicit
    SpectrumTexture(const RgbSpectrum value)
        : texture_type(TextureType::Constant), constant_spectrum{Spectrum(value)} {}

    explicit
    SpectrumTexture(const RgbSpectrumUnbounded value)
        : texture_type(TextureType::Constant), constant_spectrum{Spectrum(value)} {}

    explicit
    SpectrumTexture(const Spectrum &value)
        : texture_type(TextureType::Constant), constant_spectrum{Spectrum(value)} {}

    SpectralQuantity
    fetch(const vec2 &uv, const SampledLambdas &lambdas) const {
        switch (texture_type) {
        case TextureType::Constant:
            return constant_spectrum.eval(lambdas);
        case TextureType::Image: {
            return image_texture.fetch_spectrum(uv, lambdas);
        }
        case TextureType::Scale: {
            return scale_texture.fetch(uv, lambdas);
        }
        case TextureType::Mix:
            return mix_texture.fetch(uv, lambdas);
        default:
            panic();
        }
    }

private:
    TextureType texture_type{};
    union {
        Spectrum constant_spectrum{ConstantSpectrum(0.F)};
        ImageTexture image_texture;
        SpectrumScaleTexture scale_texture;
        SpectrumMixTexture mix_texture;
    };
};

#endif // PT_TEXTURE_H
