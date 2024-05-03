
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "texture.h"

#include <cmath>
#include <fmt/core.h>
#include <tinyexr.h>

#include "../color/spectrum.h"
#include "../utils/chunk_allocator.h"

void
transform_rgb_to_spectrum(f32 *pixels, i32 width, i32 height) {
    for (i32 p = 0; p < width * height; p++) {
        f32 r = pixels[3 * p + 0];
        f32 g = pixels[3 * p + 1];
        f32 b = pixels[3 * p + 2];

        RgbSpectrum spectrum = RgbSpectrum::make(tuple3(r, g, b));
        pixels[3 * p + 0] = spectrum.sigmoid_coeff.x;
        pixels[3 * p + 1] = spectrum.sigmoid_coeff.y;
        pixels[3 * p + 2] = spectrum.sigmoid_coeff.z;
    }
}

void
check_texture_dimensions(i32 width, i32 height) {
    if (width < 1 || height < 1 || width > 0x10'00'00 || height > 0x10'00'00) {
        throw std::runtime_error(
            fmt::format("Invalid texture dimensions: {} x {}", width, height));
    }
}

ImageTexture
load_exr_texture(const std::string &texture_path, bool is_rgb) {
    f32 *pixels = nullptr;
    i32 width = 0;
    i32 height = 0;
    u32 num_channels = 4;

    const char *err = nullptr;
    i32 ret = LoadEXR(&pixels, &width, &height, texture_path.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        if (err) {
            auto msg = fmt::format("EXR loading error: {}", err);
            FreeEXRErrorMessage(err);
            throw std::runtime_error(msg);
        }
    }

    check_texture_dimensions(width, height);

    if (is_rgb) {
        // Convert form 4 channels to 3
        f32 *pixels_3_channels =
            static_cast<f32 *>(std::malloc(width * height * 3 * sizeof(f32)));

        for (i32 p = 0; p < width * height; p++) {
            pixels_3_channels[3 * p + 0] = pixels[num_channels * p + 0] / 255.f;
            pixels_3_channels[3 * p + 1] = pixels[num_channels * p + 1] / 255.f;
            pixels_3_channels[3 * p + 2] = pixels[num_channels * p + 2] / 255.f;

            std::free(pixels);
            pixels = pixels_3_channels;

            transform_rgb_to_spectrum(pixels, width, height);
            num_channels = 3;
        }
    }

    return ImageTexture(0, 0, pixels, num_channels, TextureDataType::F32);
}

ImageTexture
load_other_format_texture(const std::string &texture_path, bool is_rgb) {
    i32 width = 0;
    i32 height = 0;
    i32 num_channels = 0;

    u8 *pixels = stbi_load(texture_path.c_str(), &width, &height, &num_channels, 0);
    check_texture_dimensions(width, height);

    if (is_rgb) {
        if (num_channels < 3 || num_channels > 4) {
            throw std::runtime_error("Invalid RGB texture channel count");
        }

        constexpr int num_channels_converted = 3;

        // Transform u8s to f32s
        f32 *pixels_f32 = static_cast<f32 *>(
            std::malloc(width * height * num_channels_converted * sizeof(f32)));

        for (i32 p = 0; p < width * height; p++) {
            pixels_f32[num_channels_converted * p + 0] =
                static_cast<f32>(pixels[num_channels * p + 0]) / 255.f;
            pixels_f32[num_channels_converted * p + 1] =
                static_cast<f32>(pixels[num_channels * p + 1]) / 255.f;
            pixels_f32[num_channels_converted * p + 2] =
                static_cast<f32>(pixels[num_channels * p + 2]) / 255.f;
        }

        stbi_image_free(pixels);
        transform_rgb_to_spectrum(pixels_f32, width, height);
        return ImageTexture(width, height, pixels_f32, num_channels_converted,
                            TextureDataType::F32);
    } else {
        if (num_channels > 4 || num_channels < 1) {
            throw std::runtime_error("Invalid texture channel count");
        }

        return ImageTexture(width, height, pixels, num_channels, TextureDataType::U8);
    }
}

ImageTexture
ImageTexture::make(const std::string &texture_path, bool is_rgb) {
    if (texture_path.ends_with(".exr")) {
        return load_exr_texture(texture_path, is_rgb);
    } else {
        return load_other_format_texture(texture_path, is_rgb);
    }
}

u64
ImageTexture::calc_index(const vec2 &uv) const {
    f32 foo;
    f32 ufrac = std::modf(uv.x, &foo);
    f32 vfrac = std::modf(uv.y, &foo);

    f32 u = ufrac < 0.f ? 1.f + ufrac : ufrac;
    f32 v = vfrac < 0.f ? 1.f + vfrac : vfrac;

    vec2 xy_sized = vec2(u, 1.f - v) * vec2(width - 1U, height - 1U);

    u32 x = xy_sized.x;
    u32 y = xy_sized.y;
    return x + (width * y);
}

Spectrum
ImageTexture::fetch_spectrum(const vec2 &uv) const {
    auto pixel_index = calc_index(uv);

    switch (data_type) {
    case TextureDataType::U8: {
        return Spectrum(ConstantSpectrum::make(1.f));
    }
    case TextureDataType::F32: {
        f32 *pixels_f32 = static_cast<f32 *>(pixels);
        // assert(num_channels == 3);

        f32 a = pixels_f32[num_channels * pixel_index];
        f32 b = pixels_f32[num_channels * pixel_index + 1];
        f32 c = pixels_f32[num_channels * pixel_index + 2];

        return Spectrum(RgbSpectrum::from_coeff(tuple3(a, b, c)));
    }
    default:
        assert(false);
    }
}

f32
ImageTexture::fetch_float(const vec2 &uv) const {
    auto pixel_index = calc_index(uv);

    switch (data_type) {
    case TextureDataType::U8: {
        u8 *pixels_u8 = static_cast<u8 *>(pixels);

        f32 a = (f32)pixels_u8[num_channels * pixel_index] / 255.f;
        return a;
    }
    case TextureDataType::F32: {
        f32 *pixels_f32 = static_cast<f32 *>(pixels);
        // assert(num_channels == 3);

        f32 a = pixels_f32[num_channels * pixel_index];

        return a;
    }
    default:
        assert(false);
    }
}
