
#include "../utils/basic_types.h"
#include "sobol_sampler.h"

#include "catch2/catch_test_macros.hpp"
#include <matplot/matplot.h>

TEST_CASE("sobol sampler basic") {
    using namespace matplot;

    std::vector<f64> x{};
    std::vector<f64> y{};

    SobolSampler s{};

    for (int i = 0; i < 256; i += 1) {
        auto xs = s.sobol_sample(i, 4);
        auto ys = s.sobol_sample(i, 5);

        x.push_back(xs);
        y.push_back(ys);
    }

    /*hold(on);
    scatter(x, y);

    // matplot::legend({std::to_string(sum)});

    show();*/
}
