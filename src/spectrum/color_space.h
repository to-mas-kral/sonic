#ifndef PT_COLOR_SPACE_H
#define PT_COLOR_SPACE_H

#include "../math/transform.h"
#include "../utils/basic_types.h"
#include "../utils/panic.h"

enum class ColorSpace : u8 {
    sRGB,
};

// clang-format off
const mat3 XYZ_TO_RGB_MATRIX = mat3::from_columns(
    tuple3(3.240812398895283F,   -0.9692430170086407F,  0.055638398436112804F),
    tuple3(-1.5373084456298136F, 1.8759663029085742F,   -0.20400746093241362F),
    tuple3(-0.4985865229069666F, 0.04155503085668564F,  1.0571295702861434F)
);
// clang-format on

static tuple3
xyz_to_srgb(const tuple3 &xyz) {
    return XYZ_TO_RGB_MATRIX * xyz;
}

static tuple3
srgb_nonlinear_to_linear(const tuple3 &rgb) {
    const auto transform = [](const f32 channel) {
        if (channel <= 0.04045F) {
            return channel / 12.92F;
        } else {
            return std::powf((channel + 0.055F) / 1.055F, 2.4F);
        }
    };

    const auto x = transform(rgb.x);
    const auto y = transform(rgb.y);
    const auto z = transform(rgb.z);

    return {x, y, z};
}

static tuple3
nonlinear_to_linear(const ColorSpace color_space, const tuple3 &rgb) {
    switch (color_space) {
    case ColorSpace::sRGB: {
        return srgb_nonlinear_to_linear(rgb);
    }
    default:
        panic();
    }
}

#endif // PT_COLOR_SPACE_H
