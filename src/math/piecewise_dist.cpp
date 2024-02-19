#include "piecewise_dist.h"

#include "../math/sampling.h"

#include <algorithm>
#include <numeric>

PiecewiseDist2D::PiecewiseDist2D(const std::vector<f32> &grid, int width, int height) {
    conditionals = std::vector<PiecewiseDist1D>(height);
    std::vector<f32> marginals_sums(height);

    for (int r = 0; r < height; r++) {
        auto row = Span<f32>(const_cast<f32 *>(&grid[r * width]), width);
        auto conditional = PiecewiseDist1D(row);
        conditionals.emplace_back(std::move(conditional));

        f32 sum = std::accumulate(row.begin(), row.end(), 0.f);
        marginals_sums.push_back(sum);
    }

    marginals = PiecewiseDist1D(Span<f32>(marginals_sums.data(), marginals_sums.size()));
}

Tuple<vec2, f32>
PiecewiseDist2D::sample(const vec2 &sample) {
    auto [v, im] = marginals.sample_continuous(sample.x);
    auto [u, ic] = conditionals[im].sample_continuous(sample.y);

    f32 pdf0 = marginals.pdf(im);
    f32 pdf1 = conditionals[im].pdf(ic);

    return {vec2(u, v), pdf0 * pdf1};
}

f32
PiecewiseDist2D::pdf(const vec2 &sample) {
    auto [pdf0, im] = marginals.sample_continuous(sample.x);
    auto [pdf1, _] = conditionals[im].sample_continuous(sample.y);

    return pdf0 * pdf1;
}

PiecewiseDist1D::PiecewiseDist1D(PiecewiseDist1D &&other) noexcept {
    this->pmf = std::move(other.pmf);
    this->cmf = std::move(other.cmf);
}

void
PiecewiseDist1D::create_cmf() {
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

PiecewiseDist1D::PiecewiseDist1D(std::vector<f32> &&p_pmf) : pmf{std::move(p_pmf)} {
    create_cmf();
}

PiecewiseDist1D::PiecewiseDist1D(Span<f32> vals) {
    pmf.reserve(vals.size());

    f32 sum = std::accumulate(vals.begin(), vals.end(), 0.f);
    for (auto v : vals) {
        pmf.push_back(v / sum);
    }

    create_cmf();
}

f32
PiecewiseDist1D::pdf(u32 index) const {
    return pmf[index];
}

u32
PiecewiseDist1D::sample(f32 sample) {
    return sample_discrete_cmf(Span<f32>(cmf.data(), cmf.size()), sample);
}

Tuple<f32, u32>
PiecewiseDist1D::sample_continuous(f32 sample) {
    return sample_continuous_cmf(Span<f32>(cmf.data(), cmf.size()), sample);
}

Tuple<f32, u32>
PiecewiseDist1D::pdf(f32 sample) const {
    u32 offset = sample * (f32)cmf.size();
    if (offset > cmf.size() - 1) {
        offset = cmf.size();
    }

    return {pmf[offset], offset};
}

PiecewiseDist1D &
PiecewiseDist1D::operator=(PiecewiseDist1D &&other) noexcept {
    this->pmf = std::move(other.pmf);
    this->cmf = std::move(other.cmf);

    return *this;
}
