#ifndef PT_TEXTURE_H
#define PT_TEXTURE_H

#include <cmath>
#include <string>

#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <tinyexr.h>

#include "../geometry/ray.h"
#include "../math/piecewise_dist.h"
#include "../math/vecmath.h"
#include "../utils/basic_types.h"
#include "../utils/chunk_allocator.h"

template <typename T> struct ConstantTexture {
    static ConstantTexture
    make(T val) {
        return ConstantTexture{
            .value = val,
        };
    }

    T
    fetch() const {
        return value;
    };

    T value;
};

enum class TextureDataType : u8 {
    U8,
    F32,
};

class ImageTexture {
public:
    ImageTexture() = default;

    ImageTexture(i32 width, i32 height, void *pixels, u32 num_channels,
                 TextureDataType data_type)
        : width{width}, height{height}, pixels{pixels}, num_channels{num_channels},
          data_type{data_type} {}

    static ImageTexture
    make(const std::string &texture_path, bool is_rgb);

    tuple3
    fetch(const vec2 &uv) const {
        f32 foo;
        f32 ufrac = std::modf(uv.x, &foo);
        f32 vfrac = std::modf(uv.y, &foo);

        f32 u = ufrac < 0.f ? 1.f + ufrac : ufrac;
        f32 v = vfrac < 0.f ? 1.f + vfrac : vfrac;

        vec2 xy_sized = vec2(u, 1.f - v) * vec2(width - 1U, height - 1U);

        u32 x = xy_sized.x;
        u32 y = xy_sized.y;
        u64 pixel_index = x + (width * y);

        // TODO: FIXME: figure out a better texture representation
        switch (data_type) {
        case TextureDataType::U8: {
            u8 *pixels_u8 = reinterpret_cast<u8 *>(pixels);
            assert(num_channels == 3);

            f32 a = (f32)pixels_u8[num_channels * pixel_index] / 255.f;
            f32 b = (f32)pixels_u8[num_channels * pixel_index + 1] / 255.f;
            f32 c = (f32)pixels_u8[num_channels * pixel_index + 2] / 255.f;

            return tuple3(a, b, c);
        }
        case TextureDataType::F32: {
            f32 *pixels_f32 = reinterpret_cast<f32 *>(pixels);
            // assert(num_channels == 3);

            f32 a = pixels_f32[num_channels * pixel_index];
            f32 b = pixels_f32[num_channels * pixel_index + 1];
            f32 c = pixels_f32[num_channels * pixel_index + 2];

            return tuple3(a, b, c);
        }
        default:
            assert(false);
        }
    };

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
    Image,
};

// TODO: templated texture ? IDK if it's a good idea
class Texture {
public:
    Texture() = default;

    static Texture
    make_image_texture(const std::string &texture_path, bool is_rgb,
                       ChunkAllocator<ImageTexture> &texture_alloc) {
        Texture tex{};
        tex.texture_type = TextureType::Image;
        tex.inner.image_texture = ImageTexture::make(texture_path, is_rgb);

        return tex;
    }

    static Texture
    make_constant_texture(f32 value) {
        Texture tex{};
        tex.texture_type = TextureType::ConstantF32;
        tex.inner.constant_texture_f32 = ConstantTexture<f32>::make(value);

        return tex;
    }

    static Texture
    make_constant_texture(RgbSpectrum value) {
        Texture tex{};
        tex.texture_type = TextureType::ConstantRgb;
        tex.inner.constant_texture_rgb = ConstantTexture<RgbSpectrum>::make(value);

        return tex;
    }

    tuple3
    fetch(const vec2 &uv) const {
        switch (texture_type) {
        case TextureType::ConstantF32:
            return tuple3(inner.constant_texture_f32.fetch());
        case TextureType::ConstantRgb:
            return inner.constant_texture_rgb.fetch().sigmoid_coeff;
        case TextureType::Image:
            return inner.image_texture.fetch(uv);
        default:
            assert(false);
        }
    };

    void
    free() const {
        if (texture_type == TextureType::Image) {
            inner.image_texture.free();
        }
    }

    TextureType texture_type{};
    union {
        ConstantTexture<f32> constant_texture_f32;
        ConstantTexture<RgbSpectrum> constant_texture_rgb;
        ImageTexture image_texture;
    } inner{};
};

#endif // PT_TEXTURE_H
