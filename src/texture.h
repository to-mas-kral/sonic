#ifndef PT_TEXTURE_H
#define PT_TEXTURE_H

#include <cmath>
#include <string>

#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <tinyexr.h>

#include "geometry/ray.h"
#include "math/piecewise_dist.h"
#include "math/vecmath.h"
#include "utils/basic_types.h"
#include "utils/chunk_allocator.h"
#include "utils/cuda_box.h"
#include "utils/cuda_err.h"
#include "utils/um_vector.h"

template <typename T> struct ConstantTexture {
    static ConstantTexture
    make(T val) {
        return ConstantTexture{
            .value = val,
        };
    }

    __device__ T
    fetch() const {
        return value;
    };

    T value;
};

class ImageTexture {
public:
    ImageTexture() = default;

    ImageTexture(cudaTextureObject_t tex_obj, cudaArray_t texture_storage_array,
                 i32 width, i32 height)
        : tex_obj(tex_obj), texture_storage_array(texture_storage_array), width(width),
          height(height) {}

    static ImageTexture
    make(const std::string &texture_path, bool is_rgb);

    __device__ tuple3
    fetch(const vec2 &uv) const {
        auto ret = tex2D<float4>(tex_obj, uv.x, 1.f - uv.y);
        return tuple3(ret.x, ret.y, ret.z);
    };

    void
    free() const {
        if (tex_obj != 0) {
            cudaDestroyTextureObject(tex_obj);
        }

        if (texture_storage_array != nullptr) {
            cudaFreeArray(texture_storage_array);
        }
    }

protected:
    cudaTextureObject_t tex_obj = 0;
    cudaArray_t texture_storage_array = nullptr;
    i32 width = 0;
    i32 height = 0;
};

enum class TextureType : u8 {
    ConstantF32,
    ConstantRgb,
    Image,
};

class Texture {
public:
    Texture() = default;

    static Texture
    make_image_texture(const std::string &texture_path, bool is_rgb,
                       UnifiedMemoryChunkAllocator<ImageTexture> &texture_alloc) {
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

    __device__ tuple3
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
