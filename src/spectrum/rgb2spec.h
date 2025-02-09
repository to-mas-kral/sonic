#ifndef PT_RGB2SPEC_H
#define PT_RGB2SPEC_H

/*
 * This is a port of https://github.com/mitsuba-renderer/rgb2spec
 * */

#include "../math/vecmath.h"
#include "../utils/basic_types.h"

#include <filesystem>
#include <vector>

constexpr u32 RGB2SPEC_N_COEFFS = 3;

struct SigmoigCoeff {
    explicit
    SigmoigCoeff(const tuple3 &inner)
        : inner(inner) {}

    tuple3 inner;
};

class RGB2Spec {
public:
    /// Load a RGB2Spec model from disk
    explicit
    RGB2Spec(const std::filesystem::path &path);

    /// Convert an RGB value into a RGB2Spec coefficient representation
    SigmoigCoeff
    fetch(const tuple3 &rgb_) const;

    static f32
    eval(const SigmoigCoeff &coeff, f32 lambda);

private:
    i32
    rgb2spec_find_interval(const f32 *values, f32 x) const;

    static f32
    rgb2spec_fma(f32 a, f32 b, f32 c);

    u32 res{};
    std::vector<f32> m_scale{};
    std::vector<f32> data{};
};

#endif // PT_RGB2SPEC_H
