#ifndef PT_PIECEWISE_DIST_H
#define PT_PIECEWISE_DIST_H

#include "../math/vecmath.h"
#include "../utils/basic_types.h"

#include <span>
#include <vector>

class PiecewiseDist1D {
public:
    PiecewiseDist1D() = default;

    PiecewiseDist1D(PiecewiseDist1D &other) = delete;

    PiecewiseDist1D(PiecewiseDist1D &&other) noexcept = default;

    PiecewiseDist1D &
    operator=(PiecewiseDist1D &other) = delete;

    PiecewiseDist1D &
    operator=(PiecewiseDist1D &&other) noexcept = default;

    /// Calculates probabilities from function values
    /// The function is defined over the range 0-1... if the domain is larger,
    /// the resulting PDF needs to be remapped (a simple lerp).
    explicit
    PiecewiseDist1D(std::span<f32> vals);

    f32
    pdf(u32 index) const;

    std::tuple<f32, u32>
    sample_continuous(f32 xi) const;

    f32
    pdf(f32 sample) const;

    u32
    size() const;

private:
    /// Cumulative distribution function
    std::vector<f32> m_cdf{};
    /// Function values
    std::vector<f32> function{};
    f32 func_int{0.f};
};

// OPTIMIZE: could reduce size by having a vector of a fixed size

/// Adapted from:
/// https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#sec:sample-discrete-2d
class PiecewiseDist2D {
public:
    explicit
    PiecewiseDist2D() = default;
    explicit
    PiecewiseDist2D(const std::vector<f32> &grid, int width, int height);

    /// Returns xy-coords (ranging 0-1) and the pdf
    ///
    /// XY coords are top to bottom!:
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
    std::tuple<vec2, f32>
    sample(const vec2 &xi) const;

    f32
    pdf(const vec2 &xy) const;

private:
    /// probability distributions in rows
    std::vector<PiecewiseDist1D> conditionals{};
    /// Probbability distributions of rows
    PiecewiseDist1D marginals{};
};

#endif // PT_PIECEWISE_DIST_H
