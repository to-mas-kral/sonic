#ifndef PT_COLOR_SPACE_H
#define PT_COLOR_SPACE_H

#include "../math/transform.h"
#include "../utils/basic_types.h"
#include "spectrum.h"

__device__ const DenseSpectrum CIE_65 = DenseSpectrum::from_static(CIE_D65_RAW);

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

#endif // PT_COLOR_SPACE_H
