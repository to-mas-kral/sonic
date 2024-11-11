#ifndef DISCRETE_DIST_H
#define DISCRETE_DIST_H

#include "../utils/basic_types.h"

#include <numeric>
#include <span>
#include <vector>

class DiscreteDist {
public:
    DiscreteDist() = default;

    DiscreteDist(DiscreteDist &other) = delete;

    DiscreteDist(DiscreteDist &&other) noexcept = default;

    DiscreteDist &
    operator=(DiscreteDist &other) = delete;

    DiscreteDist &
    operator=(DiscreteDist &&other) noexcept = default;

    ~
    DiscreteDist() = default;

    void
    create_cmf() {
        cmf.reserve(pmf.size() + 1);

        f32 cmf_sum = 0.F;
        cmf.push_back(cmf_sum);
        for (const f32 i : pmf) {
            cmf_sum += i;
            cmf.push_back(cmf_sum);
        }

        f32 err = std::abs(cmf[cmf.size() - 1] - 1.F);
        assert(err < 0.0001F);

        cmf[cmf.size() - 1] = 1.F;
    }

    /// Expects normalized probabilites !
    explicit
    DiscreteDist(std::vector<f32> &&p_pmf)
        : pmf{std::move(p_pmf)} {
        create_cmf();

        assert(!pmf.empty());
        assert(!cmf.empty());
    }

    /// Calculates probabilities
    explicit
    DiscreteDist(std::span<const f32> vals) {
        pmf.reserve(vals.size());

        const f32 sum = std::accumulate(vals.begin(), vals.end(), 0.F);
        for (const auto v : vals) {
            pmf.push_back(v / sum);
        }

        create_cmf();

        assert(!pmf.empty());
        assert(!cmf.empty());
    }

    f32
    pdf(const u32 index) const {
        return pmf[index];
    }

    /// Samples a CMF, return an index into the PMF slice.
    u32
    sample(const f32 sample) const {
        const auto i = std::upper_bound(cmf.begin(), cmf.end(), sample);
        assert(i != cmf.end());
        const auto offset = std::distance(cmf.begin(), i);
        return offset == 0 ? 0 : offset - 1;
    }

    /*std::tuple<f32, u32>
    pdf(const f32 sample) const {
        u32 offset = sample * (f32)cmf.size();
        if (offset > cmf.size() - 1) {
            offset = cmf.size();
        }

        return {pmf[offset], offset};
    }*/

private:
    /// Probability mass function
    std::vector<f32> pmf;
    /// Cumulative mass function
    std::vector<f32> cmf;
};

#endif // DISCRETE_DIST_H
