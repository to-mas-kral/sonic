#ifndef PT_PIECEWISE_DIST_H
#define PT_PIECEWISE_DIST_H

#include "../math/sampling.h"
#include "../math/vecmath.h"
#include "../utils/basic_types.h"

#include <algorithm>
#include <numeric>
#include <vector>

class PiecewiseDist1D {
public:
    PiecewiseDist1D() = default;

    PiecewiseDist1D(PiecewiseDist1D &other) = delete;

    PiecewiseDist1D(PiecewiseDist1D &&other) noexcept {
        this->pmf = std::move(other.pmf);
        this->cmf = std::move(other.cmf);
    }

    void
    create_cmf() {
        cmf.reserve(pmf.size());

        f32 cmf_sum = 0.f;
        for (f32 i : pmf) {
            cmf_sum += i;
            cmf.push_back(cmf_sum);
        }

        f32 err = std::abs(cmf[cmf.size() - 1] - 1.f);
        assert(err < 0.00001f);

        cmf[cmf.size() - 1] = 1.f;
    }

    /// Expects normalized probabilites !
    explicit PiecewiseDist1D(std::vector<f32> &&p_pmf) : pmf{std::move(p_pmf)} {
        create_cmf();
    }

    /// Calculates probabilities
    explicit PiecewiseDist1D(Span<f32> vals) {
        pmf.reserve(vals.size());

        f32 sum = std::accumulate(vals.begin(), vals.end(), 0.f);
        for (auto v : vals) {
            pmf.push_back(v / sum);
        }

        create_cmf();
    }

    f32
    pdf(u32 index) const {
        return pmf[index];
    }

    u32
    sample(f32 sample) {
        return sample_discrete_cmf(Span<f32>(cmf.data(), cmf.size()), sample);
    }

    Tuple<f32, u32>
    sample_continuous(f32 sample) {
        return sample_continuous_cmf(Span<f32>(cmf.data(), cmf.size()), sample);
    }

    Tuple<f32, u32>
    pdf(f32 sample) const {
        u32 offset = sample * (f32)cmf.size();
        if (offset > cmf.size() - 1) {
            offset = cmf.size();
        }

        return {pmf[offset], offset};
    }

    PiecewiseDist1D &
    operator=(PiecewiseDist1D &other) = delete;

    PiecewiseDist1D &
    operator=(PiecewiseDist1D &&other) noexcept {
        this->pmf = std::move(other.pmf);
        this->cmf = std::move(other.cmf);

        return *this;
    }

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
    sample(const vec2 &sample) {
        auto [v, im] = marginals.sample_continuous(sample.x);
        auto [u, ic] = conditionals[im].sample_continuous(sample.y);

        f32 pdf0 = marginals.pdf(im);
        f32 pdf1 = conditionals[im].pdf(ic);

        return {vec2(u, v), pdf0 * pdf1};
    }

    f32
    pdf(const vec2 &sample) {
        auto [pdf0, im] = marginals.sample_continuous(sample.x);
        auto [pdf1, _] = conditionals[im].sample_continuous(sample.y);

        return pdf0 * pdf1;
    }

private:
    /// probability distributions in rows
    std::vector<PiecewiseDist1D> conditionals;
    /// Probbability distributions of rows
    PiecewiseDist1D marginals;
};

#endif // PT_PIECEWISE_DIST_H
