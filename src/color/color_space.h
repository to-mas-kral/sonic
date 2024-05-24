#ifndef PT_COLOR_SPACE_H
#define PT_COLOR_SPACE_H

#include "../math/transform.h"

#include "../utils/basic_types.h"
enum class ColorSpace : u8 {
    sRGB,
};

// clang-format off
const mat3 XYZ_TO_RGB_MATRIX = mat3::from_columns(
    tuple3(3.240812398895283f,   -0.9692430170086407f,  0.055638398436112804f),
    tuple3(-1.5373084456298136f, 1.8759663029085742f,   -0.20400746093241362f),
    tuple3(-0.4985865229069666f, 0.04155503085668564f,  1.0571295702861434f)
);
// clang-format on

static tuple3
xyz_to_srgb(const tuple3 &xyz) {
    return XYZ_TO_RGB_MATRIX * xyz;
}

static tuple3
srgb_nonlinear_to_linear(const tuple3 &rgb) {
    const auto transform = [](const f32 c) {
        if (c <= 0.04045f) {
            return c / 12.92f;
        } else {
            return std::powf((c + 0.055f) / 1.055f, 2.4f);
        }
    };

    const auto x = transform(rgb.x);
    const auto y = transform(rgb.y);
    const auto z = transform(rgb.z);

    return tuple3(x, y, z);
}

static tuple3
nonlinear_to_linear(const ColorSpace color_space, const tuple3 &rgb) {
    switch (color_space) {
    case ColorSpace::sRGB: {
        return srgb_nonlinear_to_linear(rgb);
    }
    default:
        assert(false);
    }
}

#endif // PT_COLOR_SPACE_H
