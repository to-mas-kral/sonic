#include "../color/sampled_spectrum.h"
#include "../math/vecmath.h"
#include "envmap.h"

#include <catch2/catch_test_macros.hpp>

#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <random>

TEST_CASE("envmap pdf roudtrip") {
    std::mt19937 generator(73927932889);
    std::uniform_real_distribution<f32> distribution{};

    // TODO: choose smaller envmap (make a custom one)
    auto image = Image::make("../resources/test/envmap.exr");
    auto texture = ImageTexture(&image);
    auto envmap = Envmap(texture, 1.f, mat4::identity());

    i32 wrong = 0;

    for (int i = 0; i < 1000; ++i) {
        const auto x = distribution(generator);
        const auto y = distribution(generator);
        const auto rand = vec2(x, y);

        auto pos = point3(0.f);
        const auto lambdas = SampledLambdas::new_mock();

        vec3 world_dir;
        const auto sample = envmap.sample(pos, rand, lambdas, &world_dir);

        if (sample.has_value()) {
            const auto s_pdf = sample->pdf;
            const auto calc_pdf = envmap.pdf(world_dir.normalized());

            if (std::abs(s_pdf - calc_pdf) > 0.01) {
                wrong += 1;
            }
        }
    }

    // TODO: would be nice to pass this 100%, don't know what I can do about the imprecision...
    // 0.1% fail rate...
    REQUIRE((wrong < 10));
}
