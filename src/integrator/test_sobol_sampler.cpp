
#include "../utils/basic_types.h"
#include "sobol_sampler.h"

#include "catch2/catch_test_macros.hpp"

TEST_CASE("sobol sampler basic") {
    std::vector<f64> x{};
    std::vector<f64> y{};

    SobolSampler s{};

    for (int i = 0; i < 256; i += 1) {
        const auto xs = s.sobol_sample(i, 4);
        const auto ys = s.sobol_sample(i, 5);

        x.push_back(xs);
        y.push_back(ys);
    }
}
