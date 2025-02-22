#ifndef IMAGE_H
#define IMAGE_H

#include "../math/vecmath.h"
#include "../spectrum/spectrum.h"
#include "../utils/basic_types.h"

struct ImageTextureParams;

enum class ImageDataType : u8 {
    U8,
    F32,
};

/// XY - image space:
///
///  0,0 |               X
///    --.--------------->
///      |
///      |
///      |
///      |
///      |
///      |
///   Y  |
class Image {
public:
    Image(const i32 width, const i32 height, u8 *pixels, const u32 num_channels)
        : m_data_type{ImageDataType::U8}, m_num_channels{num_channels}, m_width{width},
          m_height{height}, m_pixels_u8{pixels} {}

    Image(const i32 width, const i32 height, f32 *pixels, const u32 num_channels)
        : m_data_type{ImageDataType::F32}, m_num_channels{num_channels}, m_width{width},
          m_height{height}, m_pixels_f32{pixels} {}

    static Image
    from_filepath(const std::string &texture_path);

    /// Coordinate system - see above
    tuple3
    fetch_rgb_texel(const uvec2 &coords) const;

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
    fetch_rgb_texel(const vec2 &coords) const;

    tuple3
    fetch_rgb(const vec2 &uv_in, const ImageTextureParams &params) const;

    f32
    fetch_float(const vec2 &uv_in, const ImageTextureParams &params) const;

    const ColorSpace &
    get_scolor_space() const {
        return m_color_space;
    }

    i32
    width() const {
        return m_width;
    }

    i32
    height() const {
        return m_height;
    }

    Image(const Image &other) = delete;

    Image &
    operator=(const Image &other) = delete;

    Image(Image &&other) noexcept
        : m_data_type(other.m_data_type), m_color_space(other.m_color_space),
          m_num_channels(other.m_num_channels), m_width(other.m_width),
          m_height(other.m_height) {
        if (m_data_type == ImageDataType::U8) {
            m_pixels_u8 = other.m_pixels_u8;
            other.m_pixels_u8 = nullptr;
        } else if (m_data_type == ImageDataType::F32) {
            m_pixels_f32 = other.m_pixels_f32;
            other.m_pixels_f32 = nullptr;
        }
    }

    Image &
    operator=(Image &&other) noexcept {
        if (this == &other) {
            return *this;
        }

        m_data_type = other.m_data_type;
        m_color_space = other.m_color_space;
        m_num_channels = other.m_num_channels;
        m_width = other.m_width;
        m_height = other.m_height;

        if (m_data_type == ImageDataType::U8) {
            m_pixels_u8 = other.m_pixels_u8;
            other.m_pixels_u8 = nullptr;
        } else if (m_data_type == ImageDataType::F32) {
            m_pixels_f32 = other.m_pixels_f32;
            other.m_pixels_f32 = nullptr;
        }

        return *this;
    }

    ~
    Image() {
        // Careful about UB, deallocation depends on which allocator (library...)
        // was used to allocate the memory.
        if (m_data_type == ImageDataType::U8 && m_pixels_u8 != nullptr) {
            std::free(m_pixels_u8);
        } else if (m_data_type == ImageDataType::F32 && m_pixels_f32 != nullptr) {
            std::free(m_pixels_f32);
        }
    }

private:
    u64
    calc_index(const vec2 &uv) const;

    tuple3
    rgb_from_pixel_index(u64 pixel_index) const;

    ImageDataType m_data_type{};
    ColorSpace m_color_space{};
    u32 m_num_channels{0};
    i32 m_width{0};
    i32 m_height{0};

    union {
        u8 *m_pixels_u8 = nullptr;
        f32 *m_pixels_f32;
    };
};

#endif // IMAGE_H
