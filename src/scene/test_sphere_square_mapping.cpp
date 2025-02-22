#include "../math/samplers/sampler.h"
#include "../math/vecmath.h"
#include "sphere_square_mapping.h"

#include <catch2/catch_test_macros.hpp>

#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("sphere_to_square roundtrip 1", "[sphere_to_square roundtrip 1]") {
    auto sampler = DimensionSampler();

    constexpr i32 N = 1024;
    for (int i = 0; i < N; ++i) {
        const auto uv = sampler.sample2();

        const auto vec = square_to_sphere(uv);
        const auto uv_roundtrip = sphere_to_square(vec);

        constexpr f32 MARGIN = 0.000005;
        CHECK_THAT(uv_roundtrip.x, Catch::Matchers::WithinAbs(uv.x, MARGIN));
        CHECK_THAT(uv_roundtrip.y, Catch::Matchers::WithinAbs(uv.y, MARGIN));
    }
}

TEST_CASE("sphere_to_square roundtrip 2", "[sphere_to_square roundtrip 2]") {
    auto sampler = DimensionSampler();

    constexpr i32 N = 8192;
    for (int i = 0; i < N; ++i) {
        const auto vec = sampler.sample3().normalized();

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
        const auto uv = sphere_to_square(vec3(0.1F, 0.1F, 1.F).normalized());
        CHECK(uv.approx_eq(vec2(0.52481F, 0.52481F)));
    }

    {
        const auto uv = sphere_to_square(vec3(-0.1F, 0.1F, 1.F).normalized());
        CHECK(uv.approx_eq(vec2(0.47518F, 0.52481F)));
    }

    {
        const auto uv = sphere_to_square(vec3(0.1F, -0.1F, 1.F).normalized());
        CHECK(uv.approx_eq(vec2(0.52481F, 0.47518F)));
    }

    {
        const auto uv = sphere_to_square(vec3(-0.1F, -0.1F, 1.F).normalized());
        CHECK(uv.approx_eq(vec2(0.47518F, 0.47518F)));
    }

    // Backward (-Z)

    {
        const auto uv = sphere_to_square(vec3(0.1F, 0.1F, -1.F).normalized());
        CHECK(uv.approx_eq(vec2(0.975184F, 0.975184F)));
    }

    {
        const auto uv = sphere_to_square(vec3(-0.1F, 0.1F, -1.F).normalized());
        CHECK(uv.approx_eq(vec2(0.024814F, 0.975184F)));
    }

    {
        const auto uv = sphere_to_square(vec3(0.1F, -0.1F, -1.F).normalized());
        CHECK(uv.approx_eq(vec2(0.975184F, 0.024814F)));
    }

    {
        const auto uv = sphere_to_square(vec3(-0.1F, -0.1F, -1.F).normalized());
        CHECK(uv.approx_eq(vec2(0.024814F, 0.024814F)));
    }
}
