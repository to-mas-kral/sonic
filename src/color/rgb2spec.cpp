
#include "rgb2spec.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fmt/core.h>

RGB2Spec::
RGB2Spec(const std::filesystem::path &path) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error(
            fmt::format("Couldn't find RGB2SPEC in: {}", path.c_str()));
    }

    char header[4];
    if (fread(header, 4, 1, f) != 1 || memcmp(header, "SPEC", 4) != 0) {
        fclose(f);
        throw std::runtime_error("Error while reading RGB2SPEC file");
    }

    if (fread(&res, sizeof(u32), 1, f) != 1) {
        fclose(f);
        throw std::runtime_error("Error while reading RGB2SPEC file");
    }

    const size_t size_scale = sizeof(f32) * res;
    const size_t size_data = sizeof(f32) * res * res * res * 3 * RGB2SPEC_N_COEFFS;

    m_scale.resize(size_scale);
    data.resize(size_data);

    if (fread(m_scale.data(), size_scale, 1, f) != 1 ||
        fread(data.data(), size_data, 1, f) != 1) {
        fclose(f);
        throw std::runtime_error("Error while reading RGB2SPEC file");
    }

    fclose(f);
}

tuple3
RGB2Spec::fetch(const tuple3 &rgb_) const {
    // Handle uniform values by setting c0 and c1 to zero and iverting the sigmoid
    if (rgb_[0] == rgb_[1] && rgb_[1] == rgb_[2]) {
        return tuple3(0, 0, (rgb_[0] - .5f) / std::sqrt(rgb_[0] * (1.f - rgb_[0])));
    }

    // Determine largest RGB component
    i32 i = 0;
    f32 rgb[3];

    for (i32 j = 0; j < 3; ++j) {
        rgb[j] = std::max(std::min(rgb_[j], 1.f), 0.f);
    }

    for (i32 j = 1; j < 3; ++j) {
        if (rgb[j] >= rgb[i]) {
            i = j;
        }
    }

    const f32 z = rgb[i];
    const f32 scale = (res - 1U) / z;
    const f32 x = rgb[(i + 1) % 3] * scale;
    const f32 y = rgb[(i + 2) % 3] * scale;

    // Trilinearly interpolated lookup
    const u32 xi = std::min((u32)x, (u32)(res - 2));
    const u32 yi = std::min((u32)y, (u32)(res - 2));
    const u32 zi = rgb2spec_find_interval(m_scale.data(), z);
    u32 offset = (((i * res + zi) * res + yi) * res + xi) * RGB2SPEC_N_COEFFS;
    const u32 dx = RGB2SPEC_N_COEFFS, dy = RGB2SPEC_N_COEFFS * res;
    const u32 dz = RGB2SPEC_N_COEFFS * res * res;

    const f32 x1 = x - xi;
    const f32 x0 = 1.f - x1;
    const f32 y1 = y - yi;
    const f32 y0 = 1.f - y1;
    const f32 z1 = (z - m_scale[zi]) / (m_scale[zi + 1] - m_scale[zi]);
    const f32 z0 = 1.f - z1;

    auto out = tuple3(0.f);
    for (i32 j = 0; j < RGB2SPEC_N_COEFFS; ++j) {
        out[j] = ((data[offset] * x0 + data[offset + dx] * x1) * y0 +
                  (data[offset + dy] * x0 + data[offset + dy + dx] * x1) * y1) *
                     z0 +
                 ((data[offset + dz] * x0 + data[offset + dz + dx] * x1) * y0 +
                  (data[offset + dz + dy] * x0 + data[offset + dz + dy + dx] * x1) * y1) *
                     z1;
        offset++;
    }

    return out;
}

f32
RGB2Spec::eval(const tuple3 &coeff, const f32 lambda) {
    const f32 x = rgb2spec_fma(rgb2spec_fma(coeff[0], lambda, coeff[1]), lambda, coeff[2]);
    // Handle the limit case
    if (std::isinf(x)) {
        return (x > 0.f) ? 1.f : 0.f;
    }

    const f32 y = 1.f / sqrtf(rgb2spec_fma(x, x, 1.f));
    return rgb2spec_fma(.5f * x, y, .5f);
}

i32
RGB2Spec::rgb2spec_find_interval(const f32 *values, const f32 x) const {
    i32 left = 0;
    const i32 last_interval = res - 2;
    i32 size = last_interval;

    while (size > 0) {
        const i32 half = size >> 1;
        const i32 middle = left + half + 1;

        if (values[middle] <= x) {
            left = middle;
            size -= half + 1;
        } else {
            size = half;
        }
    }

    return std::min(left, last_interval);
}

f32
RGB2Spec::rgb2spec_fma(const f32 a, const f32 b, const f32 c) {
    return a * b + c;
}
