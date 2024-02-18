
#include "piecewise_dist.h"

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
