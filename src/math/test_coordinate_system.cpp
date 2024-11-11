
#include <catch2/catch_test_macros.hpp>

#include "../integrator/shading_frame.h"

#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <random>

TEST_CASE("shading frame basis vector e2e") {
    const auto normal = vec3(0.5F, 0.2F, 0.7F).normalized();
    const auto sframe = CoordinateSystem(normal);

    REQUIRE(sframe.to_local(normal).approx_eq(vec3(0.F, 0.F, 1.F)));
    REQUIRE(sframe.from_local(norm_vec3(0.F, 0.F, 1.F)).approx_eq(normal));
}

TEST_CASE("shading frame e2e") {
    std::mt19937 rng(73927932889); // NOLINT(*-msc51-cpp)

    for (int n = 0; n < 64; ++n) {
        const auto nx = std::generate_canonical<f32, 23>(rng);
        const auto ny = std::generate_canonical<f32, 23>(rng);
        const auto nz = std::generate_canonical<f32, 23>(rng);

        const auto normal = vec3(nx, ny, nz).normalized();
        const auto normal_local = norm_vec3(0.F, 0.F, 1.F);

        const auto sframe = CoordinateSystem(normal);

        for (int v = 0; v < 64; ++v) {
            const auto vx = std::generate_canonical<f32, 23>(rng);
            const auto vy = std::generate_canonical<f32, 23>(rng);
            const auto vz = std::generate_canonical<f32, 23>(rng);

            const auto vec = vec3(vx, vy, vz).normalized();

            const auto local = sframe.to_local(vec).normalized();
            const auto back = sframe.from_local(local);

            REQUIRE(vec.approx_eq(back));
            REQUIRE_THAT(
                vec3::dot(normal, vec),
                Catch::Matchers::WithinAbs(vec3::dot(normal_local, local), 0.00001));
        }
    }
}
