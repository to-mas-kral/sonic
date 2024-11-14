
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "texture.h"

#include <algorithm>
#include <cmath>
#include <fmt/core.h>
#include <tinyexr.h>

namespace {
void
check_texture_dimensions(i32 width, i32 height) {
    if (width < 1 || height < 1 || width > 0x10'00'00 || height > 0x10'00'00) {
        throw std::runtime_error(
            fmt::format("Invalid texture dimensions: {} x {}", width, height));
    }
}

Image
load_exr_image(const std::string &texture_path) {
    f32 *pixels = nullptr;
    i32 width = 0;
    i32 height = 0;
    constexpr u32 num_channels = 4;

    const char *err = nullptr;
    const auto ret = LoadEXR(&pixels, &width, &height, texture_path.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        if (err != nullptr) {
            const auto msg = fmt::format("EXR loading error: {}", err);
            FreeEXRErrorMessage(err);
            throw std::runtime_error(msg);
        }
    }

    check_texture_dimensions(width, height);

    return Image(width, height, pixels, num_channels);
}

Image
load_other_format_image(const std::string &texture_path) {
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

vec2
calc_uv(const vec2 &uv, const ImageTextureParams &params) {
    return uv * vec2(params.uscale, params.vscale) + vec2(params.udelta, params.vdelta);
}
} // namespace

Image
Image::from_filepath(const std::string &texture_path) {
    if (texture_path.ends_with(".exr")) {
        return load_exr_image(texture_path);
    } else {
        return load_other_format_image(texture_path);
    }
}

tuple3
Image::fetch_rgb_texel(const uvec2 &coords) const {
    assert(coords.x >= 0 && coords.x < width);
    assert(coords.y >= 0 && coords.y < height);

    const auto pixel_index = coords.x + (m_width * coords.y);
    return rgb_from_pixel_index(pixel_index);
}

tuple3
Image::fetch_rgb_texel(const vec2 &coords) const {
    assert(coords.x >= 0.F && coords.x <= 1.F);
    assert(coords.y >= 0.F && coords.y <= 1.F);

    auto xy_integer = uvec2(coords.x * m_width, coords.y * m_height);

    if (xy_integer.x == m_width) {
        xy_integer.x = m_width - 1;
    }

    if (xy_integer.y == m_height) {
        xy_integer.y = m_height - 1;
    }

    return fetch_rgb_texel(xy_integer);
}

u64
Image::calc_index(const vec2 &uv) const {
    f32 foo;
    const f32 ufrac = std::modf(uv.x, &foo);
    const f32 vfrac = std::modf(uv.y, &foo);

    const f32 u = ufrac < 0.F ? 1.F + ufrac : ufrac;
    const f32 v = vfrac < 0.F ? 1.F + vfrac : vfrac;

    const vec2 xy_integer = vec2(u, v) * vec2(m_width, m_height);

    u32 x = xy_integer.x;
    u32 y = xy_integer.y;

    if (x >= m_width) {
        x = m_width - 1;
    }

    if (y >= m_height) {
        y = m_height - 1;
    }

    return x + (m_width * y);
}

tuple3
Image::rgb_from_pixel_index(const u64 pixel_index) const {
    switch (m_data_type) {
    case ImageDataType::U8: {
        if (m_num_channels >= 3) {
            const auto r =
                static_cast<f32>(m_pixels_u8[pixel_index * m_num_channels]) / 255.F;
            const auto g =
                static_cast<f32>(m_pixels_u8[pixel_index * m_num_channels + 1]) / 255.F;
            const auto b =
                static_cast<f32>(m_pixels_u8[pixel_index * m_num_channels + 2]) / 255.F;
            return nonlinear_to_linear(m_color_space, tuple3(r, g, b));
        } else {
            const auto g =
                static_cast<f32>(m_pixels_u8[pixel_index * m_num_channels]) / 255.F;
            return nonlinear_to_linear(m_color_space, tuple3(g, g, g));
        }
    }
    case ImageDataType::F32: {
        if (m_num_channels >= 3) {
            const auto r = m_pixels_f32[pixel_index * m_num_channels];
            const auto g = m_pixels_f32[pixel_index * m_num_channels + 1];
            const auto b = m_pixels_f32[pixel_index * m_num_channels + 2];
            return tuple3(r, g, b);
        } else {
            const auto g = m_pixels_f32[pixel_index * m_num_channels];
            return tuple3(g, g, g);
        }
    }
    default:
        panic();
    }
}

tuple3
Image::fetch_rgb(const vec2 &uv_in, const ImageTextureParams &params) const {
    const auto uv = calc_uv(uv_in, params);

    const auto pixel_index = calc_index(uv);
    assert(pixel_index < static_cast<i64>(width) * static_cast<i64>(height));

    auto scaled = rgb_from_pixel_index(pixel_index) * params.scale;
    if (params.invert) {
        scaled = (tuple3(1.F) - scaled).clamp_negative();
    }

    return scaled;
}

f32
Image::fetch_float(const vec2 &uv_in, const ImageTextureParams &params) const {
    const auto uv = calc_uv(uv_in, params);

    const auto pixel_index = calc_index(uv);
    assert(pixel_index < static_cast<i64>(width) * static_cast<i64>(height));

    f32 fetched = 0.F;

    switch (m_data_type) {
    case ImageDataType::U8: {
        if (m_num_channels == 4) {
            fetched =
                static_cast<f32>(m_pixels_u8[pixel_index * m_num_channels + 3]) / 255.F;
        } else {
            fetched = static_cast<f32>(m_pixels_u8[pixel_index * m_num_channels]) / 255.F;
        }
        break;
    }
    case ImageDataType::F32: {
        if (m_num_channels == 4) {
            fetched = m_pixels_f32[pixel_index * m_num_channels + 3];
        } else {
            fetched = m_pixels_f32[pixel_index * m_num_channels];
        }
        break;
    }
    default:
        panic();
    }

    fetched *= params.scale;
    if (params.invert) {
        fetched = 1.F - fetched;
        fetched = std::max(fetched, 0.F);
    }

    return fetched;
}
