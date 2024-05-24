#ifndef IMAGE_H
#define IMAGE_H

#include "../color/spectrum.h"
#include "../math/vecmath.h"
#include "../utils/basic_types.h"

enum class ImageDataType : u8 {
    U8,
    F32,
};

class Image {
public:
    Image(const i32 width, const i32 height, u8 *pixels, const u32 num_channels)
        : data_type{ImageDataType::U8}, num_channels{num_channels}, width{width},
          height{height}, pixels_u8{pixels} {}

    Image(const i32 width, const i32 height, f32 *pixels, const u32 num_channels)
        : data_type{ImageDataType::F32}, num_channels{num_channels}, width{width},
          height{height}, pixels_f32{pixels} {}

    static Image
    make(const std::string &texture_path);

    tuple3
    fetch_rgb_texel(const uvec2 &coords) const;

    tuple3
    fetch_rgb(const vec2 &uv) const;

    f32
    fetch_float(const vec2 &uv) const;

    const ColorSpace &
    get_scolor_space() const {
        return color_space;
    }

    i32
    get_width() const {
        return width;
    }

    i32
    get_height() const {
        return height;
    }

    void
    free() const {
        if (data_type == ImageDataType::U8) {
            delete[] pixels_u8;
        } else if (data_type == ImageDataType::F32) {
            delete[] pixels_f32;
        }
    }

private:
    u64
    calc_index(const vec2 &uv) const;

    tuple3
    get_rgb_pixel_index(u64 pixel_index) const;

    ImageDataType data_type{};
    ColorSpace color_space{};
    u32 num_channels{0};
    i32 width{0};
    i32 height{0};

    union {
        u8 *pixels_u8 = nullptr;
        f32 *pixels_f32;
    };
};

#endif // IMAGE_H
