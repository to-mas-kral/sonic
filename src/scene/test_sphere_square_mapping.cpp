#include "../math/vecmath.h"
#include "sphere_square_mapping.h"

#include <catch2/catch_test_macros.hpp>

#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <random>

TEST_CASE("sphere_to_square roundtrip 1", "[sphere_to_square roundtrip 1]") {
    std::mt19937 generator(73927932889);
    std::uniform_real_distribution<f32> distribution{};

    for (int i = 0; i < 1000; ++i) {
        const auto x = distribution(generator);
        const auto y = distribution(generator);
        const auto uv = vec2(x, y);

        const auto vec = square_to_sphere(uv);
        const auto uv_roundtrip = sphere_to_square(vec);

        REQUIRE_THAT(uv_roundtrip.x, Catch::Matchers::WithinAbs(uv.x, 0.001));
        REQUIRE_THAT(uv_roundtrip.y, Catch::Matchers::WithinAbs(uv.y, 0.001));
    }
}

TEST_CASE("sphere_to_square roundtrip 2", "[sphere_to_square roundtrip 2]") {
    std::mt19937 generator(73927932889);
    std::uniform_real_distribution<f32> distribution{};

    for (int i = 0; i < 1000; ++i) {
        const auto x = distribution(generator);
        const auto y = distribution(generator);
        const auto z = distribution(generator);
        const auto vec = vec3(x, y, z).normalized();

        const auto uv = sphere_to_square(vec);
        const auto vec_roundtrip = square_to_sphere(uv);

        CHECK_THAT(vec_roundtrip.x, Catch::Matchers::WithinAbs(vec.x, 0.001));
        CHECK_THAT(vec_roundtrip.y, Catch::Matchers::WithinAbs(vec.y, 0.001));
        CHECK_THAT(vec_roundtrip.z, Catch::Matchers::WithinAbs(vec.z, 0.001));
    }
}

