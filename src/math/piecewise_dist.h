#ifndef PT_PIECEWISE_DIST_H
#define PT_PIECEWISE_DIST_H

#include <numeric>

#include "../math/sampling.h"
#include "../utils/basic_types.h"
#include "../utils/um_vector.h"

class PiecewiseDist1D {
public:
    void
    create_cmf() {
        cmf = UmVector<f32>(pmf.size());

        f32 cmf_sum = 0.f;
        for (int i = 0; i < pmf.size(); i++) {
            cmf_sum += pmf[i];
            cmf.push(cmf_sum);
        }

        f32 err = abs(cmf[cmf.size() - 1] - 1.f);
        assert(err < 0.00001f);

        cmf[cmf.size() - 1] = 1.f;
    }

    PiecewiseDist1D() = default;

    /// Expects normalized probabilites !
    explicit PiecewiseDist1D(UmVector<f32> &&p_pmf) : pmf{std::move(p_pmf)} {
        create_cmf();
    }

    /// Calculates probabilities
    explicit PiecewiseDist1D(CSpan<f32> vals) {
        pmf = UmVector<f32>(vals.size());

        f32 sum = std::accumulate(vals.begin(), vals.end(), 0.f);
        for (auto v : vals) {
            pmf.push(v / sum);
        }

        create_cmf();
    }

    __device__ __forceinline__ f32
    pdf(u32 index) const {
        return pmf[index];
    }

    __device__ __forceinline__ u32
    sample(f32 sample) const {
        return sample_discrete_cmf(CSpan<f32>(cmf.get_ptr(), cmf.size()), sample);
    }

    __device__ __forceinline__ CTuple<f32, u32>
    sample_continuous(f32 sample) const {
        return sample_continuous_cmf(CSpan<f32>(cmf.get_ptr(), cmf.size()), sample);
    }

    __device__ __forceinline__ CTuple<f32, u32>
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
    UmVector<f32> pmf{};
    /// Cumulative mass function
    UmVector<f32> cmf{};
};

// OPTIMIZE: could reduce size by having a UmVector with a fixed size

/// Adapted from:
/// https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#sec:sample-discrete-2d
class PiecewiseDist2D {
public:
    explicit PiecewiseDist2D() = default;
    explicit PiecewiseDist2D(const UmVector<f32> &grid, int width, int height) {
        conditionals = UmVector<PiecewiseDist1D>(height);
        UmVector<f32> marginals_sums(height);

        for (int r = 0; r < height; r++) {
            auto row = CSpan<f32>(const_cast<f32 *>(&grid[r * width]), width);
            conditionals.push(PiecewiseDist1D(row));

            f32 sum = std::accumulate(row.begin(), row.end(), 0.f);
            marginals_sums.push(sum);
        }

        marginals = PiecewiseDist1D(CSpan<f32>(
            const_cast<f32 *>(marginals_sums.get_ptr()), marginals_sums.size()));
    }

    /// Returns uv-coords and the pdf
    __device__ __forceinline__ CTuple<vec2, f32>
    sample(const vec2 &sample) const {
        auto [v, im] = marginals.sample_continuous(sample.x);
        auto [u, ic] = conditionals[im].sample_continuous(sample.y);

        f32 pdf0 = marginals.pdf(im);
        f32 pdf1 = conditionals[im].pdf(ic);

        return {vec2(u, v), pdf0 * pdf1};
    }

    __device__ __forceinline__ f32
    pdf(const vec2 &sample) const {
        auto [pdf0, im] = marginals.sample_continuous(sample.x);
        auto [pdf1, _] = conditionals[im].sample_continuous(sample.y);

        return pdf0 * pdf1;
    }

private:
    /// probability distributions in rows
    UmVector<PiecewiseDist1D> conditionals;
    /// Probbability distributions of rows
    PiecewiseDist1D marginals;
};

#endif // PT_PIECEWISE_DIST_H
