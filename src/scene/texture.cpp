
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "../color/spectrum.h"
#include "texture.h"

void
transform_rgb_to_spectrum(f32 *pixels, i32 width, i32 height) {
    for (i32 p = 0; p < width * height; p++) {
        f32 r = pixels[4 * p + 0];
        f32 g = pixels[4 * p + 1];
        f32 b = pixels[4 * p + 2];

        RgbSpectrum spectrum = RgbSpectrum::make(tuple3(r, g, b));
        pixels[4 * p + 0] = spectrum.sigmoid_coeff.x;
        pixels[4 * p + 1] = spectrum.sigmoid_coeff.y;
        pixels[4 * p + 2] = spectrum.sigmoid_coeff.z;
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
    constexpr u32 num_channels = 4;

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
        transform_rgb_to_spectrum(pixels, width, height);
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

        constexpr int num_channels_converted = 4;

        // Transform u8s to f32s
        f32 *pixels_f32 = reinterpret_cast<f32 *>(
            std::malloc(width * height * num_channels_converted * sizeof(f32)));

        for (i32 p = 0; p < width * height; p++) {
            pixels_f32[4 * p + 0] =
                static_cast<f32>(pixels[num_channels * p + 0]) / 255.f;
            pixels_f32[4 * p + 1] =
                static_cast<f32>(pixels[num_channels * p + 1]) / 255.f;
            pixels_f32[4 * p + 2] =
                static_cast<f32>(pixels[num_channels * p + 2]) / 255.f;
            pixels_f32[4 * p + 3] = 1.f;
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
