#ifndef PT_RGB2SPEC_H
#define PT_RGB2SPEC_H

/*
 * This is a port of https://github.com/mitsuba-renderer/rgb2spec
 * */

#include "../math/vecmath.h"
#include "../utils/basic_types.h"

#include <filesystem>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/core.h>
#include <vector>

constexpr u32 RGB2SPEC_N_COEFFS = 3;

class RGB2Spec {
public:
    /// Load a RGB2Spec model from disk
    explicit RGB2Spec(const std::filesystem::path &path) {
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

        size_t size_scale = sizeof(f32) * res;
        size_t size_data = sizeof(f32) * res * res * res * 3 * RGB2SPEC_N_COEFFS;

        m_scale.resize(size_scale);
        data.resize(size_data);

        if (fread(m_scale.data(), size_scale, 1, f) != 1 ||
            fread(data.data(), size_data, 1, f) != 1) {
            fclose(f);
            throw std::runtime_error("Error while reading RGB2SPEC file");
        }

        fclose(f);
    }
    /// Convert an RGB value into a RGB2Spec coefficient representation
    tuple3
    fetch(const tuple3 &rgb_) const {
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

        f32 z = rgb[i];
        f32 scale = (res - 1U) / z;
        f32 x = rgb[(i + 1) % 3] * scale;
        f32 y = rgb[(i + 2) % 3] * scale;

        // Trilinearly interpolated lookup
        u32 xi = std::min((u32)x, (u32)(res - 2));
        u32 yi = std::min((u32)y, (u32)(res - 2));
        u32 zi = rgb2spec_find_interval(m_scale.data(), z);
        u32 offset = (((i * res + zi) * res + yi) * res + xi) * RGB2SPEC_N_COEFFS;
        u32 dx = RGB2SPEC_N_COEFFS, dy = RGB2SPEC_N_COEFFS * res;
        u32 dz = RGB2SPEC_N_COEFFS * res * res;

        f32 x1 = x - xi;
        f32 x0 = 1.f - x1;
        f32 y1 = y - yi;
        f32 y0 = 1.f - y1;
        f32 z1 = (z - m_scale[zi]) / (m_scale[zi + 1] - m_scale[zi]);
        f32 z0 = 1.f - z1;

        auto out = tuple3(0.f);
        for (i32 j = 0; j < RGB2SPEC_N_COEFFS; ++j) {
            out[j] =
                ((data[offset] * x0 + data[offset + dx] * x1) * y0 +
                 (data[offset + dy] * x0 + data[offset + dy + dx] * x1) * y1) *
                    z0 +
                ((data[offset + dz] * x0 + data[offset + dz + dx] * x1) * y0 +
                 (data[offset + dz + dy] * x0 + data[offset + dz + dy + dx] * x1) * y1) *
                    z1;
            offset++;
        }

        return out;
    }

    __host__ __device__ static f32
    eval(const tuple3 &coeff, f32 lambda) {
        f32 x = rgb2spec_fma(rgb2spec_fma(coeff[0], lambda, coeff[1]), lambda, coeff[2]);
        // Handle the limit case
        if (isinf(x)) {
            return (x > 0.f) ? 1.f : 0.f;
        }

        f32 y = 1.f / sqrtf(rgb2spec_fma(x, x, 1.f));
        return rgb2spec_fma(.5f * x, y, .5f);
    }

private:
    i32
    rgb2spec_find_interval(const f32 *values, f32 x) const {
        i32 left = 0;
        i32 last_interval = res - 2;
        i32 size = last_interval;

        while (size > 0) {
            i32 half = size >> 1, middle = left + half + 1;

            if (values[middle] <= x) {
                left = middle;
                size -= half + 1;
            } else {
                size = half;
            }
        }

        return std::min(left, last_interval);
    }

    __host__ __device__ static f32
    rgb2spec_fma(f32 a, f32 b, f32 c) {
        return a * b + c;
    }

    u32 res;
    std::vector<f32> m_scale{};
    std::vector<f32> data{};
};

#endif // PT_RGB2SPEC_H
