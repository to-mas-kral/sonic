#include "../scene/envmap.h"
#include "../test/test_globals.h"
#include "discrete_dist.h"
#include "piecewise_dist.h"
#include "samplers/sampler.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>
#include <random>

TEST_CASE("sample discrete dist") {
    //                    0 0.2  0.5  0.8  0.85  1.
    std::vector<f32> pmf = {0.2F, 0.3F, 0.3F, 0.05F, 0.15F};
    const auto dist = DiscreteDist(pmf);

    {
        const auto index = dist.sample(0.1F);
        REQUIRE(index == 0);
        REQUIRE(dist.pdf(index) == 0.2F);
    }

    {
        const auto index = dist.sample(0.3F);
        REQUIRE(index == 1);
        REQUIRE(dist.pdf(index) == 0.3F);
    }

    {
        const auto index = dist.sample(0.6F);
        REQUIRE(index == 2);
        REQUIRE(dist.pdf(index) == 0.3F);
    }

    {
        const auto index = dist.sample(0.84F);
        REQUIRE(index == 3);
        REQUIRE(dist.pdf(index) == 0.05F);
    }

    {
        const auto index = dist.sample(0.99F);
        REQUIRE(index == 4);
        REQUIRE(dist.pdf(index) == 0.15F);
    }
}

TEST_CASE("sample piecewise1d constant") {
    std::vector<f32> func = {2.F, 2.F};
    const auto dist = PiecewiseDist1D(func);

    {
        const auto [res, index] = dist.sample_continuous(0.2F);
        REQUIRE(index == 0);
        REQUIRE(dist.pdf(index) == 1.F);
        CHECK_THAT(res, Catch::Matchers::WithinAbs(0.2F, 0.00001F));
    }
}

TEST_CASE("sample piecewise1d") {
    std::vector<f32> func = {1.F, 2.F};
    const auto dist = PiecewiseDist1D(func);

    {
        const auto [res, index] = dist.sample_continuous(0.2F);
        REQUIRE(index == 0);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(1.F / 1.5F, 0.00001F));
    }

    {
        const auto [res, index] = dist.sample_continuous(0.4);
        REQUIRE(index == 1);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(2.F / 1.5F, 0.00001F));
    }
}

TEST_CASE("sample piecewise1d second") {
    std::vector<f32> func = {2.F, 4.F, 1.F, 3.F, 2.F};
    const auto dist = PiecewiseDist1D(func);

    constexpr auto integral = 12.F / 5.F;

    {
        const auto [res, index] = dist.sample_continuous(0.05F);
        REQUIRE(index == 0);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(2.F / integral, 0.00001F));
    }

    {
        const auto [res, index] = dist.sample_continuous(0.25F);
        REQUIRE(index == 1);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(4.F / integral, 0.00001F));
    }

    {
        const auto [res, index] = dist.sample_continuous(0.54F);
        REQUIRE(index == 2);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(1.F / integral, 0.00001F));
    }

    {
        const auto [res, index] = dist.sample_continuous(0.66F);
        REQUIRE(index == 3);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(3.F / integral, 0.00001F));
    }

    {
        const auto [res, index] = dist.sample_continuous(0.92F);
        REQUIRE(index == 4);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(2.F / integral, 0.00001F));
    }
}

TEST_CASE("sample piecewise1d third") {
    std::vector<f32> func = {1.F, 2.F, 4.F, 2.F, 1.F};
    const auto dist = PiecewiseDist1D(func);

    constexpr auto integral = 10.F / 5.F;

    {
        const auto [res, index] = dist.sample_continuous(0.F);
        REQUIRE(index == 0);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(1.F / integral, 0.00001F));
        CHECK_THAT(dist.pdf(res), Catch::Matchers::WithinAbs(1.F / integral, 0.00001F));
        CHECK_THAT(res, Catch::Matchers::WithinAbs(0.F, 0.00001F));
    }

    {
        const auto [res, index] = dist.sample_continuous(0.1F);
        REQUIRE(index == 1);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(2.F / integral, 0.00001F));
        CHECK_THAT(dist.pdf(res), Catch::Matchers::WithinAbs(2.F / integral, 0.00001F));
        CHECK_THAT(res, Catch::Matchers::WithinAbs(0.2F, 0.00001F));
    }

    {
        const auto [res, index] = dist.sample_continuous(0.5f);
        REQUIRE(index == 2);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(4.F / integral, 0.00001F));
        CHECK_THAT(dist.pdf(res), Catch::Matchers::WithinAbs(4.F / integral, 0.00001F));
        CHECK_THAT(res, Catch::Matchers::WithinAbs(0.5F, 0.00001F));
    }

    {
        const auto [res, index] = dist.sample_continuous(0.75f);
        REQUIRE(index == 3);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(2.F / integral, 0.00001F));
        CHECK_THAT(dist.pdf(res), Catch::Matchers::WithinAbs(2.F / integral, 0.00001F));
        CHECK_THAT(res, Catch::Matchers::WithinAbs(0.65F, 0.00001F));
    }

    {
        const auto [res, index] = dist.sample_continuous(0.95F);
        REQUIRE(index == 4);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(1.F / integral, 0.00001F));
        CHECK_THAT(dist.pdf(res), Catch::Matchers::WithinAbs(1.F / integral, 0.00001F));
        CHECK_THAT(res, Catch::Matchers::WithinAbs(0.9F, 0.00001F));
    }
}

TEST_CASE("piecewise2d pdf roundtrip") {
    std::vector<f32> func = {};

    constexpr i32 SIZE = 4096;
    func.reserve(SIZE * SIZE);

    auto sampler = DimensionSampler();
    std::mt19937 generator(73927932889);
    std::normal_distribution norm{1.0, 100.0};

    for (i32 i = 0; i < SIZE * SIZE; ++i) {
        func.emplace_back(std::abs(norm(generator)));
    }

    const auto dist = PiecewiseDist2D::from_grid(func, SIZE, SIZE);

    constexpr i32 N = 1024;
    for (i32 i = 0; i < N; ++i) {
        const auto xi = sampler.sample2();

        const auto [xy, sample_pdf] = dist.sample(xi);
        const auto pdf = dist.pdf(xy);

        CHECK_THAT(pdf, Catch::Matchers::WithinRel(sample_pdf, 0.001F));
    }
}

TEST_CASE("piecewise2d pdf roundtrip extreme values") {
    constexpr i32 SIZE = 4096;
    std::vector<f32> func(SIZE * SIZE, 0.F);

    func[10000] = 1000.F;
    func[200000] = 10000000.F;
    func[300000] = 20000000.F;
    func[400000] = 300000000.F;
    func[500000] = 5000000000.F;

    auto sampler = DimensionSampler();

    const auto dist = PiecewiseDist2D::from_grid(func, SIZE, SIZE);

    constexpr i32 N = 1024;
    for (i32 i = 0; i < N; ++i) {
        const auto xi = sampler.sample2();

        const auto [xy, sample_pdf] = dist.sample(xi);
        const auto pdf = dist.pdf(xy);

        CHECK_THAT(pdf, Catch::Matchers::WithinRel(sample_pdf, 0.001F));
    }
}

TEST_CASE("piecewise2d pdf roundtrip image") {
    auto tex = ImageTexture(&sonic::ENVMAP_BIG_TEST_IMAGE.value());

    std::vector<f32> sampling_grid(tex.width() * tex.height(), 0.F);

    const auto lambdas = SampledLambdas::sample_uniform(0.4F);

    spectral sum_rad = spectral::ZERO();

    for (int x = 0; x < tex.width(); ++x) {
        for (int y = 0; y < tex.height(); ++y) {
            const auto rgb = tex.fetch_rgb_texel(uvec2(x, y));
            const auto spec_illum = RgbSpectrumIlluminant(rgb, ColorSpace::sRGB);
            const auto spec = Spectrum(spec_illum);
            const auto rad = spec.eval(lambdas);

            sampling_grid[x + (tex.width() * y)] = (rgb.x + rgb.y + rgb.z) / 3.F;
            sum_rad += rad;
        }
    }

    auto dist = PiecewiseDist2D::from_grid(sampling_grid, tex.width(), tex.height());

    auto sampler = DimensionSampler();

    constexpr i32 N = 1024;
    for (i32 i = 0; i < N; ++i) {
        const auto xi = sampler.sample2();

        const auto [xy, sample_pdf] = dist.sample(xi);
        const auto pdf = dist.pdf(xy);

        CHECK_THAT(pdf, Catch::Matchers::WithinRel(sample_pdf, 0.001F));
    }
}

TEST_CASE("piecewise2d sampling") {
    // clang-format off
    const std::vector<f32> func = {
        0.F, 0.F, 0.F, 0.F, 0.F,
        0.F, 0.F, 0.F, 1.F, 0.F,
        0.F, 0.F, 0.F, 0.F, 0.F,
        0.F, 0.F, 0.F, 0.F, 0.F,
        0.F, 0.F, 0.F, 0.F, 0.F,
    };
    // clang-format on

    const auto dist = PiecewiseDist2D::from_grid(func, 5, 5);

    auto sampler = DimensionSampler();

    constexpr i32 N = 1024;
    for (i32 i = 0; i < N; ++i) {
        const auto xi = sampler.sample2();
        const auto [uv, pdf] = dist.sample(xi);

        REQUIRE_THAT(pdf, Catch::Matchers::WithinAbs(25.F, 0.0001F));

        REQUIRE_THAT(uv.x, Catch::Matchers::WithinAbs(0.7F, 0.1F));
        REQUIRE_THAT(uv.y, Catch::Matchers::WithinAbs(0.3F, 0.1F));
    }
}

TEST_CASE("piecewise1d pdf integrates to 1") {
    std::vector<f32> func = {1.F, 2.F, 4.F, 2.F, 1.F};
    const auto dist = PiecewiseDist1D(func);

    auto sampler = DimensionSampler();

    f32 sumpdf = 0.F;

    constexpr i32 N = 8192;

    for (i32 i = 0; i < N; ++i) {
        const auto xi = sampler.sample();
        sumpdf += dist.pdf(xi);
    }

    f32 integral = (1.F / N) * sumpdf;

    REQUIRE_THAT(integral, Catch::Matchers::WithinRel(1.F, 0.01F));
}

TEST_CASE("piecewise2d pdf integrates to 1") {
    // clang-format off
    const std::vector<f32> func = {
        1.F, 2.F, 4.F, 2.F,
        1.5F, 8.F, 4.F, 2.8F,
        17.F, 2.F, 3.F, 21.F,
        1.F, 3.2F, 1.4F, 5.F
    };
    // clang-format on

    const auto dist = PiecewiseDist2D::from_grid(func, 4, 4);

    auto sampler = DimensionSampler();

    f32 sumpdf = 0.F;

    constexpr i32 N = 8192;

    for (i32 i = 0; i < N; ++i) {
        sumpdf += dist.pdf(sampler.sample2());
    }

    f32 integral = (1.F / N) * sumpdf;

    REQUIRE_THAT(integral, Catch::Matchers::WithinRel(1.F, 0.01F));
}
