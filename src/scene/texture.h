#ifndef PT_TEXTURE_H
#define PT_TEXTURE_H

#include <string>

#include <spdlog/spdlog.h>

#include "../color/spectrum.h"
#include "../math/vecmath.h"
#include "../utils/basic_types.h"

enum class TextureDataType : u8 {
    U8,
    F32,
};

// TODO: Texture code is all over the place, currently assuming 3 and 4-channel
// textures... are 2 and 4-channel textures even needed ??
class ImageTexture {
public:
    ImageTexture(i32 width, i32 height, void *pixels, u32 num_channels,
                 TextureDataType data_type)
        : width{width}, height{height}, pixels{pixels}, num_channels{num_channels},
          data_type{data_type} {}

    static ImageTexture
    make(const std::string &texture_path, bool is_rgb);

    u64
    calc_index(const vec2 &uv) const;

    Spectrum
    fetch_spectrum(const vec2 &uv) const;

    f32
    fetch_float(const vec2 &uv) const;

    void
    free() const {
        std::free(pixels);
    }

protected:
    i32 width = 0;
    i32 height = 0;
    void *pixels;
    u32 num_channels;
    TextureDataType data_type;
};

enum class TextureType : u8 {
    ConstantF32,
    ConstantRgb,
    ConstantRgbUnbounded,
    ConstantSpectrum,
    Image,
};

class Texture {
public:
    Texture() = default;

    static Texture
    make_image_texture(const std::string &texture_path, bool is_rgb) {
        Texture tex{};
        tex.texture_type = TextureType::Image;
        tex.inner.image_texture = ImageTexture::make(texture_path, is_rgb);

        return tex;
    }

    static Texture
    make_constant_texture(f32 value) {
        Texture tex{};
        tex.texture_type = TextureType::ConstantF32;
        tex.inner.constant_texture_f32 = value;

        return tex;
    }

    static Texture
    make_constant_texture(RgbSpectrum value) {
        Texture tex{};
        tex.texture_type = TextureType::ConstantRgb;
        tex.inner.constant_texture_rgb = value;

        return tex;
    }

    static Texture
    make_constant_texture(RgbSpectrumUnbounded value) {
        Texture tex{};
        tex.texture_type = TextureType::ConstantRgb;
        tex.inner.constant_texture_rgb_unbounded = value;

        return tex;
    }

    static Texture
    make_constant_texture(Spectrum value) {
        Texture tex{};
        tex.texture_type = TextureType::ConstantSpectrum;
        tex.inner.constant_texture_spectrum = value;

        return tex;
    }

    // Some of these fetches might not be necessarily correct, but that's not worth
    // checking on every texture access in release...

    Spectrum
    fetch_spectrum(const vec2 &uv) const {
        switch (texture_type) {
        case TextureType::ConstantF32:
            return Spectrum(ConstantSpectrum::make(inner.constant_texture_f32));
        case TextureType::ConstantRgb:
            return Spectrum(inner.constant_texture_rgb);
        case TextureType::Image:
            return inner.image_texture.fetch_spectrum(uv);
        case TextureType::ConstantRgbUnbounded:
            return Spectrum(inner.constant_texture_rgb_unbounded);
        case TextureType::ConstantSpectrum:
            return inner.constant_texture_spectrum;
        default:
            assert(false);
        }
    }

    f32
    fetch_float(const vec2 &uv) const {
        switch (texture_type) {
        case TextureType::ConstantF32:
            return inner.constant_texture_f32;
        case TextureType::ConstantRgb:
            assert(false);
            return inner.constant_texture_rgb.sigmoid_coeff.x;
        case TextureType::Image:
            return inner.image_texture.fetch_float(uv);
        case TextureType::ConstantRgbUnbounded:
            assert(false);
            return inner.constant_texture_rgb_unbounded.sigmoid_coeff.x;
        case TextureType::ConstantSpectrum:
            assert(false);
            return inner.constant_texture_spectrum.eval_single(400.f);
        default:
            assert(false);
        }
    }

    void
    free() const {
        if (texture_type == TextureType::Image) {
            inner.image_texture.free();
        }
    }

    TextureType texture_type{};
    union {
        f32 constant_texture_f32;
        RgbSpectrum constant_texture_rgb;
        RgbSpectrumUnbounded constant_texture_rgb_unbounded;
        Spectrum constant_texture_spectrum;
        ImageTexture image_texture;
    } inner{};
};

#endif // PT_TEXTURE_H
