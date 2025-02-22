#include "../math/sampling.h"
#include "../math/vecmath.h"
#include "../spectrum/spectral_quantity.h"
#include "../test/test_globals.h"
#include "envmap.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("envmap pdf roudtrip") {
    auto sampler = DimensionSampler();

    auto texture = ImageTexture(&sonic::ENVMAP_BIG_TEST_IMAGE.value());
    auto envmap = Envmap::from_image(texture, 1.F, mat4::identity());

    i32 wrong = 0;

    constexpr int N = 2048;

    for (int i = 0; i < N; ++i) {
        const auto rand = sampler.sample2();

        auto pos = point3(0.F);
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
    auto sampler = DimensionSampler();

    auto texture = ImageTexture(&sonic::ENVMAP_BIG_TEST_IMAGE.value());
    auto envmap = Envmap::from_image(texture, 1.F, mat4::identity());

    i32 wrong = 0;

    constexpr int N = 1024;

    for (int i = 0; i < N; ++i) {
        const auto rand = sampler.sample2();

        auto pos = point3(0.F);
        const auto lambdas = SampledLambdas::new_mock();

        norm_vec3 world_dir;
        const auto sample = envmap.sample(pos, rand, lambdas, &world_dir);

        if (sample.has_value()) {
            const auto rad = envmap.get_ray_radiance(
                Ray(point3(0.F), world_dir.normalized()), lambdas);

            if (std::abs((rad - sample->emission).max_component()) > 0.01) {
                wrong += 1;
            }
        }
    }

    REQUIRE(((f32)wrong / (f32)N) < 0.005);
}

TEST_CASE("envmap emission pdf discrepancy") {
    auto sampler = DimensionSampler();

    auto texture = ImageTexture(&sonic::ENVMAP_BIG_TEST_IMAGE.value());
    auto envmap = Envmap::from_image(texture, 1.F, mat4::identity());

    for (int i = 0; i < 1024; ++i) {
        const auto rand = sampler.sample2();

        const auto lambdas = SampledLambdas::new_mock();
        const auto dir = sample_uniform_sphere(rand);

        const auto pdf = envmap.pdf(dir);
        const auto rad = envmap.get_ray_radiance(Ray(point3(0.F), dir), lambdas);

        CHECK(std::abs(rad.max_component() - pdf) < 1000.F);
    }
}

TEST_CASE("envmap pdf integrates to 1") {
    auto sampler = DimensionSampler();

    auto image = Image::from_filepath("../resources/test/envmap_test.exr");
    const auto texture = ImageTexture(&image);
    const auto envmap = Envmap::from_image(texture, 1.F, mat4::identity());

    f32 sumpdf = 0.F;

    constexpr i32 N = 2 * 8192;

    for (i32 i = 0; i < N; ++i) {
        const auto xi_dir = sample_uniform_sphere(sampler.sample2());

        const auto pdf = envmap.pdf(xi_dir);

        sumpdf += pdf;
    }

    f32 integral = (4.F * M_PIf) * (1.F / N) * sumpdf;

    REQUIRE_THAT(integral, Catch::Matchers::WithinRel(1.F, 0.01F));
}
