#include "../math/vecmath.h"
#include "sphere_square_mapping.h"

#include <catch2/catch_test_macros.hpp>

#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <random>

TEST_CASE("sphere_to_square roundtrip 1", "[sphere_to_square roundtrip 1]") {
    std::mt19937 generator(73927932889);
    std::uniform_real_distribution<f32> distribution{};

    constexpr i32 N = 8192;
    for (int i = 0; i < N; ++i) {
        const auto x = distribution(generator);
        const auto y = distribution(generator);
        const auto uv = vec2(x, y);

        const auto vec = square_to_sphere(uv);
        const auto uv_roundtrip = sphere_to_square(vec);

        constexpr f32 MARGIN = 0.000005;
        CHECK_THAT(uv_roundtrip.x, Catch::Matchers::WithinAbs(uv.x, MARGIN));
        CHECK_THAT(uv_roundtrip.y, Catch::Matchers::WithinAbs(uv.y, MARGIN));
    }
}

TEST_CASE("sphere_to_square roundtrip 2", "[sphere_to_square roundtrip 2]") {
    std::mt19937 generator(73927932889);
    std::uniform_real_distribution<f32> distribution{};

    constexpr i32 N = 8192;
    for (int i = 0; i < N; ++i) {
        const auto x = distribution(generator);
        const auto y = distribution(generator);
        const auto z = distribution(generator);
        const auto vec = vec3(x, y, z).normalized();

        const auto uv = sphere_to_square(vec);
        const auto vec_roundtrip = square_to_sphere(uv);

        constexpr f32 MARGIN = 0.00005;
        CHECK_THAT(vec_roundtrip.x, Catch::Matchers::WithinAbs(vec.x, MARGIN));
        CHECK_THAT(vec_roundtrip.y, Catch::Matchers::WithinAbs(vec.y, MARGIN));
        CHECK_THAT(vec_roundtrip.z, Catch::Matchers::WithinAbs(vec.z, MARGIN));
    }
}

TEST_CASE("sphere_to_square mapping") {
    // Forward (+Z)

    {
        const auto uv = sphere_to_square(vec3(0.1f, 0.1f, 1.f).normalized());
        CHECK(uv.approx_eq(vec2(0.52481f, 0.52481f)));
    }

    {
        const auto uv = sphere_to_square(vec3(-0.1f, 0.1f, 1.f).normalized());
        CHECK(uv.approx_eq(vec2(0.47518f, 0.52481f)));
    }

    {
        const auto uv = sphere_to_square(vec3(0.1f, -0.1f, 1.f).normalized());
        CHECK(uv.approx_eq(vec2(0.52481f, 0.47518f)));
    }

    {
        const auto uv = sphere_to_square(vec3(-0.1f, -0.1f, 1.f).normalized());
        CHECK(uv.approx_eq(vec2(0.47518f, 0.47518f)));
    }

    // Backward (-Z)

    {
        const auto uv = sphere_to_square(vec3(0.1f, 0.1f, -1.f).normalized());
        CHECK(uv.approx_eq(vec2(0.975184f, 0.975184f)));
    }

    {
        const auto uv = sphere_to_square(vec3(-0.1f, 0.1f, -1.f).normalized());
        CHECK(uv.approx_eq(vec2(0.024814f, 0.975184f)));
    }

    {
        const auto uv = sphere_to_square(vec3(0.1f, -0.1f, -1.f).normalized());
        CHECK(uv.approx_eq(vec2(0.975184f, 0.024814f)));
    }

    {
        const auto uv = sphere_to_square(vec3(-0.1f, -0.1f, -1.f).normalized());
        CHECK(uv.approx_eq(vec2(0.024814f, 0.024814f)));
    }
}
