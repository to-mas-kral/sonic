#include "piecewise_dist.h"

#include "../math/sampling.h"

#include <algorithm>
#include <numeric>
#include <spdlog/spdlog.h>

PiecewiseDist2D::
PiecewiseDist2D(const std::vector<f32> &grid, int width, int height) {
    conditionals.reserve(height);
    std::vector<f32> marginals_sums{};
    marginals_sums.reserve(height);

    for (int r = 0; r < height; r++) {
        auto row = std::span<f32>(const_cast<f32 *>(&grid[r * width]), width);
        auto conditional = PiecewiseDist1D(row);
        conditionals.emplace_back(std::move(conditional));

        f32 sum = std::accumulate(row.begin(), row.end(), 0.f);
        marginals_sums.push_back(sum);
    }

    marginals =
        PiecewiseDist1D(std::span<f32>(marginals_sums.data(), marginals_sums.size()));
}

std::tuple<vec2, f32>
PiecewiseDist2D::sample(const vec2 &sample) const {
    auto [v, im] = marginals.sample_continuous(sample.x);
    auto [u, ic] = conditionals[im].sample_continuous(sample.y);

    f32 pdf0 = marginals.pdf(im);
    f32 pdf1 = conditionals[im].pdf(ic);

    return {vec2(u, v), pdf0 * pdf1};
}

f32
PiecewiseDist2D::pdf(const vec2 &sample) const {
    const auto pdf0 = marginals.pdf(sample.y);

    const auto im = sample.y * marginals.size();
    const auto pdf1 = conditionals[im].pdf(sample.x);

    return pdf0 * pdf1;
}

/// Code taken from PBRTv4
PiecewiseDist1D::
PiecewiseDist1D(std::span<f32> vals) {
    function.assign(vals.begin(), vals.end());
    m_cdf = std::vector<f32>(vals.size() + 1);

    m_cdf[0] = 0;
    const auto n = vals.size();
    for (size_t i = 1; i < n + 1; ++i) {
        m_cdf[i] = m_cdf[i - 1] + vals[i - 1] / n;
    }

    func_int = m_cdf[n];
    if (func_int == 0) {
        for (size_t i = 1; i < n + 1; ++i) {
            m_cdf[i] = static_cast<f32>(i) / static_cast<f32>(n);
        }
    } else {
        for (size_t i = 1; i < n + 1; ++i) {
            m_cdf[i] /= func_int;
        }
    }

    assert(!m_cdf.empty());
}

f32
PiecewiseDist1D::pdf(const u32 index) const {
    if (func_int == 0.f) {
        return 0.f;
    }
    return function[index] / func_int;
}

f32
PiecewiseDist1D::pdf(const f32 sample) const {
    if (func_int == 0.f) {
        return 0.f;
    }

    u32 offset = (u32)(sample * (f32)function.size());
    if (offset > m_cdf.size() - 1) {
        offset = m_cdf.size();
    }

    return function[offset] / func_int;
}

u32
PiecewiseDist1D::size() const {
    return function.size();
}

/// Samples a CMF, return a value in [0, 1), and an index into the PDF slice.
std::tuple<f32, u32>
PiecewiseDist1D::sample_continuous(const f32 sample) const {
    const auto i = std::upper_bound(m_cdf.begin(), m_cdf.end(), sample);
    assert(i != m_cdf.end());
    auto offset = std::distance(m_cdf.begin(), i);
    offset--;
    assert(offset < m_cdf.size());

    f32 du = sample - m_cdf[offset];
    if ((m_cdf[offset + 1] - m_cdf[offset]) > 0) {
        du /= (m_cdf[offset + 1] - m_cdf[offset]);
    }

    f32 res = (offset + du) / (f32)(m_cdf.size() - 1u);

    return {res, offset};
}
