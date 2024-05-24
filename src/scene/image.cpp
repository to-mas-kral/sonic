
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cmath>
#include <fmt/core.h>
#include <tinyexr.h>

#include "../utils/chunk_allocator.h"

void
check_texture_dimensions(i32 width, i32 height) {
    if (width < 1 || height < 1 || width > 0x10'00'00 || height > 0x10'00'00) {
        throw std::runtime_error(
            fmt::format("Invalid texture dimensions: {} x {}", width, height));
    }
}

Image
load_exr_texture(const std::string &texture_path) {
    f32 *pixels = nullptr;
    i32 width = 0;
    i32 height = 0;
    u32 num_channels = 4;

    const char *err = nullptr;
    const auto ret = LoadEXR(&pixels, &width, &height, texture_path.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        if (err) {
            const auto msg = fmt::format("EXR loading error: {}", err);
            FreeEXRErrorMessage(err);
            throw std::runtime_error(msg);
        }
    }

    check_texture_dimensions(width, height);

    return Image(width, height, pixels, num_channels);
}

Image
load_other_format_texture(const std::string &texture_path) {
    i32 width = 0;
    i32 height = 0;
    i32 num_channels = 0;

    u8 *pixels = stbi_load(texture_path.c_str(), &width, &height, &num_channels, 0);
    check_texture_dimensions(width, height);

    if (num_channels < 1 || num_channels > 4) {
        stbi_image_free(pixels);
        throw std::runtime_error("Invalid texture channel count");
    }

    return Image(width, height, pixels, num_channels);
}

Image
Image::make(const std::string &texture_path) {
    if (texture_path.ends_with(".exr")) {
        return load_exr_texture(texture_path);
    } else {
        return load_other_format_texture(texture_path);
    }
}

tuple3
Image::fetch_rgb_texel(const uvec2 &coords) const {
    const auto pixel_index = coords.x + (width * coords.y);
    return get_rgb_pixel_index(pixel_index);
}

u64
Image::calc_index(const vec2 &uv) const {
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

tuple3
Image::get_rgb_pixel_index(const u64 pixel_index) const {
    switch (data_type) {
    case ImageDataType::U8: {
        if (num_channels >= 3) {
            const auto r =
                static_cast<f32>(pixels_u8[pixel_index * num_channels]) / 255.f;
            const auto g =
                static_cast<f32>(pixels_u8[pixel_index * num_channels + 1]) / 255.f;
            const auto b =
                static_cast<f32>(pixels_u8[pixel_index * num_channels + 2]) / 255.f;
            return nonlinear_to_linear(color_space, tuple3(r, g, b));
        } else {
            const auto g =
                static_cast<f32>(pixels_u8[pixel_index * num_channels]) / 255.f;
            return nonlinear_to_linear(color_space, tuple3(g, g, g));
        }
    }
    case ImageDataType::F32: {
        if (num_channels >= 3) {
            const auto r = pixels_f32[pixel_index * num_channels];
            const auto g = pixels_f32[pixel_index * num_channels + 1];
            const auto b = pixels_f32[pixel_index * num_channels + 2];
            return tuple3(r, g, b);
        } else {
            const auto g = pixels_f32[pixel_index * num_channels];
            return tuple3(g, g, g);
        }
    }
    default:
        assert(false);
    }
}

tuple3
Image::fetch_rgb(const vec2 &uv) const {
    const auto pixel_index = calc_index(uv);
    assert(pixel_index < static_cast<i64>(width) * static_cast<i64>(height));

    return get_rgb_pixel_index(pixel_index);
}

f32
Image::fetch_float(const vec2 &uv) const {
    const auto pixel_index = calc_index(uv);
    assert(pixel_index < static_cast<i64>(width) * static_cast<i64>(height));

    switch (data_type) {
    case ImageDataType::U8: {
        if (num_channels == 4) {
            return static_cast<f32>(pixels_u8[pixel_index * num_channels + 3]) / 255.f;
        } else {
            return static_cast<f32>(pixels_u8[pixel_index * num_channels]) / 255.f;
        }
    }
    case ImageDataType::F32: {
        if (num_channels == 4) {
            return pixels_f32[pixel_index * num_channels + 3];
        } else {
            return pixels_f32[pixel_index * num_channels];
        }
    }
    default:
        assert(false);
    }
}
