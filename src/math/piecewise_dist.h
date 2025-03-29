#ifndef PT_PIECEWISE_DIST_H
#define PT_PIECEWISE_DIST_H

#include "../math/vecmath.h"
#include "../utils/basic_types.h"

#include <numeric>
#include <span>
#include <vector>

class PiecewiseDist1D {
public:
    PiecewiseDist1D(PiecewiseDist1D const &other) = default;

    PiecewiseDist1D(PiecewiseDist1D &&other) noexcept = default;

    PiecewiseDist1D &
    operator=(PiecewiseDist1D const &other) = default;

    PiecewiseDist1D &
    operator=(PiecewiseDist1D &&other) noexcept = default;

    ~
    PiecewiseDist1D() = default;

    /// Calculates probabilities from function values
    /// The function is defined over the range 0-1... if the domain is larger,
    /// the resulting PDF needs to be remapped (a simple lerp).
    explicit
    PiecewiseDist1D(std::span<const f32> vals);

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
    std::vector<f32> m_cdf;
    /// Function values
    std::vector<f32> function;
    f32 func_int{0.F};
};

/// Adapted from:
/// https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#sec:sample-discrete-2d
class PiecewiseDist2D {
public:
    PiecewiseDist2D(PiecewiseDist2D &other) = delete;

    PiecewiseDist2D(PiecewiseDist2D &&other) noexcept = default;

    PiecewiseDist2D &
    operator=(PiecewiseDist2D &other) = delete;

    PiecewiseDist2D &
    operator=(PiecewiseDist2D &&other) noexcept = default;

    ~
    PiecewiseDist2D() = default;

    static PiecewiseDist2D
    from_grid(const std::vector<f32> &grid, int width, int height);

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
    explicit
    PiecewiseDist2D(std::vector<PiecewiseDist1D> &&conditionals,
                    PiecewiseDist1D &&marginals)
        : conditionals(std::move(conditionals)), marginals(std::move(marginals)) {}

    /// probability distributions in rows
    std::vector<PiecewiseDist1D> conditionals;
    /// Probbability distributions of rows
    PiecewiseDist1D marginals;
};

#endif // PT_PIECEWISE_DIST_H
