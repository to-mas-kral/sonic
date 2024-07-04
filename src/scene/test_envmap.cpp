#include "../color/sampled_spectrum.h"
#include "../math/sampling.h"
#include "../math/vecmath.h"
#include "envmap.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <random>

TEST_CASE("envmap pdf roudtrip") {
    std::mt19937 generator(73927932889);
    std::uniform_real_distribution<f32> distribution{};

    auto image = Image::make("../resources/test/abandoned_tank_farm_03_4k.exr");
    auto texture = ImageTexture(&image);
    auto envmap = Envmap(texture, 1.f, mat4::identity());

    i32 wrong = 0;

    constexpr int N = 8192;

    for (int i = 0; i < N; ++i) {
        const auto x = distribution(generator);
        const auto y = distribution(generator);
        const auto rand = vec2(x, y);

        auto pos = point3(0.f);
        const auto lambdas = SampledLambdas::new_mock();

        norm_vec3 world_dir;
        vec2 uv;
        const auto sample = envmap.sample(pos, rand, lambdas, &world_dir, &uv);

        if (sample.has_value()) {
            const auto s_pdf = sample->pdf;
            const auto calc_pdf = envmap.pdf(world_dir);

            if (std::abs(s_pdf - calc_pdf) > 0.01) {
                wrong += 1;
            }
        }
    }

    // Not too sure what I can do about this inaccuracy...
    REQUIRE(((f32)wrong / (f32)N) < 0.005);
}

TEST_CASE("envmap emission roudtrip") {
    std::mt19937 generator(73927932889);
    std::uniform_real_distribution<f32> distribution{};

    auto image = Image::make("../resources/test/abandoned_tank_farm_03_4k.exr");
    auto texture = ImageTexture(&image);
    auto envmap = Envmap(texture, 1.f, mat4::identity());

    i32 wrong = 0;

    constexpr int N = 8192;

    for (int i = 0; i < N; ++i) {
        const auto x = distribution(generator);
        const auto y = distribution(generator);
        const auto rand = vec2(x, y);

        auto pos = point3(0.f);
        const auto lambdas = SampledLambdas::new_mock();

        norm_vec3 world_dir;
        const auto sample = envmap.sample(pos, rand, lambdas, &world_dir);

        if (sample.has_value()) {
            const auto rad = envmap.get_ray_radiance(
                Ray(point3(0.f), world_dir.normalized()), lambdas);

            if (std::abs((rad - sample->emission).max_component()) > 0.01) {
                wrong += 1;
            }
        }
    }

    REQUIRE(((f32)wrong / (f32)N) < 0.005);
}

TEST_CASE("envmap emission pdf discrepancy") {
    std::mt19937 generator(73927932889);
    std::uniform_real_distribution<f32> distribution{};

    auto image = Image::make("../resources/test/abandoned_tank_farm_03_4k.exr");
    auto texture = ImageTexture(&image);
    auto envmap = Envmap(texture, 1.f, mat4::identity());

    for (int i = 0; i < 8192; ++i) {
        const auto x = distribution(generator);
        const auto y = distribution(generator);
        const auto rand = vec2(x, y);

        const auto lambdas = SampledLambdas::new_mock();
        const auto dir = sample_uniform_sphere(rand);

        const auto pdf = envmap.pdf(dir);
        const auto rad = envmap.get_ray_radiance(Ray(point3(0.f), dir), lambdas);

        CHECK(std::abs(rad.max_component() - pdf) < 1000.f);
    }
}

TEST_CASE("envmap pdf integrates to 1") {
    std::mt19937 generator(73927932889);
    std::uniform_real_distribution<f32> distribution{};

    auto image = Image::make("../resources/test/envmap_test.exr");
    auto texture = ImageTexture(&image);
    auto envmap = Envmap(texture, 1.f, mat4::identity());

    f32 sumpdf = 0.f;

    constexpr i32 N = 8192 * 4;

    for (i32 i = 0; i < N; ++i) {
        const auto xi_dir =
            sample_uniform_sphere(vec2(distribution(generator), distribution(generator)));

        const auto pdf = envmap.pdf(xi_dir);

        sumpdf += pdf;
    }

    f32 integral = (4.f * M_PIf) * (1.f / N) * sumpdf;

    REQUIRE_THAT(integral, Catch::Matchers::WithinRel(1.f, 0.01f));
}
