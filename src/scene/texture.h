#ifndef PT_TEXTURE_H
#define PT_TEXTURE_H

#include "image.h"

#include "../color/spectrum.h"
#include "../math/vecmath.h"
#include "../utils/basic_types.h"

class ImageTexture {
public:
    explicit
    ImageTexture(Image *image)
        : image{image} {}

    tuple3
    fetch_rgb(const vec2 &uv) const {
        return image->fetch_rgb(uv);
    }

    f32
    fetch_float(const vec2 &uv) const {
        return image->fetch_float(uv);
    }

    const ColorSpace &
    color_space() const {
        return image->get_scolor_space();
    }

private:
    Image *image;
};

enum class TextureType : u8 {
    Constant,
    Image,
    Scale,
};

class FloatTexture;

class FloatScaleTexture {
public:
    FloatScaleTexture(FloatTexture *texture, const f32 scale)
        : texture{texture}, scale{scale} {}

    f32
    fetch(const vec2 &uv) const;

private:
    FloatTexture *texture{nullptr};
    f32 scale{1.f};
};

class FloatTexture {
public:
    FloatTexture() = default;

    static FloatTexture
    make(const ImageTexture image_texture) {
        FloatTexture tex{};
        tex.texture_type = TextureType::Image;
        tex.inner.image = image_texture;

        return tex;
    }

    static FloatTexture
    make(FloatTexture *texture, const f32 scale) {
        FloatTexture tex{};
        tex.texture_type = TextureType::Scale;
        tex.inner.scale_texture = FloatScaleTexture{texture, scale};

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
        default:
            assert(false);
        }
    }

    TextureType texture_type{};
    union {
        f32 constant_f32;
        ImageTexture image;
        FloatScaleTexture scale_texture;
    } inner{};
};

enum class TextureSpectrumType : u8 {
    Rgb,
    Unbounded,
    Illuminant,
};

class SpectrumTexture;

class SpectrumScaleTexture {
public:
    SpectrumScaleTexture(SpectrumTexture *texture, const f32 scale)
        : texture{texture}, scale{scale} {}

    Spectrum
    fetch(const vec2 &uv) const;

private:
    SpectrumTexture *texture{nullptr};
    f32 scale{1.f};
};

class SpectrumTexture {
public:
    SpectrumTexture() = default;

    static SpectrumTexture
    make(const ImageTexture image_texture, const TextureSpectrumType spectrum_type) {
        SpectrumTexture tex{};
        tex.spectrum_type = spectrum_type;
        tex.texture_type = TextureType::Image;
        tex.inner.image = image_texture;

        return tex;
    }

    static SpectrumTexture
    make(SpectrumTexture *texture, const f32 scale) {
        SpectrumTexture tex{};
        tex.spectrum_type = texture->spectrum_type;
        tex.texture_type = TextureType::Scale;
        tex.inner.scale_texture = SpectrumScaleTexture{texture, scale};

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

    Spectrum
    fetch(const vec2 &uv, const f32 scale = 1.f) const {
        switch (texture_type) {
        case TextureType::Constant:
            return inner.constant_spectrum;
        case TextureType::Image: {
            const auto rgb = inner.image.fetch_rgb(uv);

            switch (spectrum_type) {
            case TextureSpectrumType::Rgb:
                return Spectrum(RgbSpectrum::make(rgb));
            case TextureSpectrumType::Unbounded:
                return Spectrum(RgbSpectrumUnbounded::make(rgb));
            case TextureSpectrumType::Illuminant:
                return Spectrum(
                    RgbSpectrumIlluminant::make(rgb, inner.image.color_space()));
            default:
                assert(false);
            }
        }
        case TextureType::Scale: {
            return inner.scale_texture.fetch(uv);
        }
        default:
            assert(false);
        }
    }

    TextureSpectrumType spectrum_type{};
    TextureType texture_type{};
    union {
        Spectrum constant_spectrum{ConstantSpectrum::make(0.f)};
        ImageTexture image;
        SpectrumScaleTexture scale_texture;
    } inner{};
};

#endif // PT_TEXTURE_H
