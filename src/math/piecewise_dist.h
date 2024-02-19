#ifndef PT_PIECEWISE_DIST_H
#define PT_PIECEWISE_DIST_H

#include "../math/vecmath.h"
#include "../utils/basic_types.h"

#include <vector>

class PiecewiseDist1D {
public:
    PiecewiseDist1D() = default;

    PiecewiseDist1D(PiecewiseDist1D &other) = delete;

    PiecewiseDist1D(PiecewiseDist1D &&other) noexcept;

    void
    create_cmf();

    /// Expects normalized probabilites !
    explicit PiecewiseDist1D(std::vector<f32> &&p_pmf);

    /// Calculates probabilities
    explicit PiecewiseDist1D(Span<f32> vals);

    f32
    pdf(u32 index) const;

    u32
    sample(f32 sample);

    Tuple<f32, u32>
    sample_continuous(f32 sample);

    Tuple<f32, u32>
    pdf(f32 sample) const;

    PiecewiseDist1D &
    operator=(PiecewiseDist1D &other) = delete;

    PiecewiseDist1D &
    operator=(PiecewiseDist1D &&other) noexcept;

private:
    /// Probability mass function
    std::vector<f32> pmf{};
    /// Cumulative mass function
    std::vector<f32> cmf{};
};

// OPTIMIZE: could reduce size by having a vector of a fixed size

/// Adapted from:
/// https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#sec:sample-discrete-2d
class PiecewiseDist2D {
public:
    explicit PiecewiseDist2D() = default;
    explicit PiecewiseDist2D(const std::vector<f32> &grid, int width, int height);

    /// Returns uv-coords and the pdf
    Tuple<vec2, f32>
    sample(const vec2 &sample);

    f32
    pdf(const vec2 &sample);

private:
    /// probability distributions in rows
    std::vector<PiecewiseDist1D> conditionals;
    /// Probbability distributions of rows
    PiecewiseDist1D marginals;
};

#endif // PT_PIECEWISE_DIST_H
