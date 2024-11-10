#include "../scene/envmap.h"
#include "discrete_dist.h"
#include "piecewise_dist.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>
#include <random>

TEST_CASE("sample discrete dist") {
    //                    0 0.2  0.5  0.8  0.85  1.
    std::vector<f32> pmf = {0.2, 0.3, 0.3, 0.05, 0.15};
    const auto dist = DiscreteDist(pmf);

    {
        const auto index = dist.sample(0.1f);
        REQUIRE(index == 0);
        REQUIRE(dist.pdf(index) == 0.2f);
    }

    {
        const auto index = dist.sample(0.3f);
        REQUIRE(index == 1);
        REQUIRE(dist.pdf(index) == 0.3f);
    }

    {
        const auto index = dist.sample(0.6f);
        REQUIRE(index == 2);
        REQUIRE(dist.pdf(index) == 0.3f);
    }

    {
        const auto index = dist.sample(0.84f);
        REQUIRE(index == 3);
        REQUIRE(dist.pdf(index) == 0.05f);
    }

    {
        const auto index = dist.sample(0.99f);
        REQUIRE(index == 4);
        REQUIRE(dist.pdf(index) == 0.15f);
    }
}

TEST_CASE("sample piecewise1d constant") {
    std::vector<f32> func = {2.f, 2.f};
    const auto dist = PiecewiseDist1D(func);

    {
        const auto [res, index] = dist.sample_continuous(0.2);
        REQUIRE(index == 0);
        REQUIRE(dist.pdf(index) == 1.f);
        CHECK_THAT(res, Catch::Matchers::WithinAbs(0.2f, 0.00001));
    }
}

TEST_CASE("sample piecewise1d") {
    std::vector<f32> func = {1.f, 2.f};
    const auto dist = PiecewiseDist1D(func);

    {
        const auto [res, index] = dist.sample_continuous(0.2);
        REQUIRE(index == 0);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(1.f / 1.5f, 0.00001));
    }

    {
        const auto [res, index] = dist.sample_continuous(0.4);
        REQUIRE(index == 1);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(2.f / 1.5f, 0.00001));
    }
}

TEST_CASE("sample piecewise1d second") {
    std::vector<f32> func = {2.f, 4.f, 1.f, 3.f, 2.f};
    const auto dist = PiecewiseDist1D(func);

    constexpr auto integral = 12.f / 5.f;

    {
        const auto [res, index] = dist.sample_continuous(0.05);
        REQUIRE(index == 0);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(2.f / integral, 0.00001));
    }

    {
        const auto [res, index] = dist.sample_continuous(0.25);
        REQUIRE(index == 1);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(4.f / integral, 0.00001));
    }

    {
        const auto [res, index] = dist.sample_continuous(0.54);
        REQUIRE(index == 2);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(1.f / integral, 0.00001));
    }

    {
        const auto [res, index] = dist.sample_continuous(0.66);
        REQUIRE(index == 3);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(3.f / integral, 0.00001));
    }

    {
        const auto [res, index] = dist.sample_continuous(0.92);
        REQUIRE(index == 4);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(2.f / integral, 0.00001));
    }
}

TEST_CASE("sample piecewise1d third") {
    std::vector<f32> func = {1.f, 2.f, 4.f, 2.f, 1.f};
    const auto dist = PiecewiseDist1D(func);

    constexpr auto integral = 10.f / 5.f;

    {
        const auto [res, index] = dist.sample_continuous(0.f);
        REQUIRE(index == 0);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(1.f / integral, 0.00001));
        CHECK_THAT(dist.pdf(res), Catch::Matchers::WithinAbs(1.f / integral, 0.00001));
        CHECK_THAT(res, Catch::Matchers::WithinAbs(0.f, 0.00001));
    }

    {
        const auto [res, index] = dist.sample_continuous(0.1f);
        REQUIRE(index == 1);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(2.f / integral, 0.00001));
        CHECK_THAT(dist.pdf(res), Catch::Matchers::WithinAbs(2.f / integral, 0.00001));
        CHECK_THAT(res, Catch::Matchers::WithinAbs(0.2f, 0.00001));
    }

    {
        const auto [res, index] = dist.sample_continuous(0.5f);
        REQUIRE(index == 2);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(4.f / integral, 0.00001));
        CHECK_THAT(dist.pdf(res), Catch::Matchers::WithinAbs(4.f / integral, 0.00001));
        CHECK_THAT(res, Catch::Matchers::WithinAbs(0.5f, 0.00001));
    }

    {
        const auto [res, index] = dist.sample_continuous(0.75f);
        REQUIRE(index == 3);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(2.f / integral, 0.00001));
        CHECK_THAT(dist.pdf(res), Catch::Matchers::WithinAbs(2.f / integral, 0.00001));
        CHECK_THAT(res, Catch::Matchers::WithinAbs(0.65f, 0.00001));
    }

    {
        const auto [res, index] = dist.sample_continuous(0.95f);
        REQUIRE(index == 4);
        CHECK_THAT(dist.pdf(index), Catch::Matchers::WithinAbs(1.f / integral, 0.00001));
        CHECK_THAT(dist.pdf(res), Catch::Matchers::WithinAbs(1.f / integral, 0.00001));
        CHECK_THAT(res, Catch::Matchers::WithinAbs(0.9f, 0.00001));
    }
}

TEST_CASE("piecewise2d pdf roundtrip") {
    std::vector<f32> func = {};

    constexpr i32 SIZE = 4096;
    func.reserve(SIZE * SIZE);

    std::mt19937 generator(73927932889);
    std::uniform_real_distribution<f32> distribution{};

    std::normal_distribution norm{1.0, 100.0};

    for (i32 i = 0; i < SIZE * SIZE; ++i) {
        func.emplace_back(std::abs(norm(generator)));
    }

    const auto dist = PiecewiseDist2D(func, SIZE, SIZE);

    constexpr i32 N = 32768;
    for (i32 i = 0; i < N; ++i) {
        const auto a = distribution(generator);
        const auto b = distribution(generator);
        const auto xi = vec2(a, b);

        const auto [xy, sample_pdf] = dist.sample(xi);
        const auto pdf = dist.pdf(xy);

        CHECK_THAT(pdf, Catch::Matchers::WithinRel(sample_pdf, 0.001f));
    }
}

TEST_CASE("piecewise2d pdf roundtrip extreme values") {
    constexpr i32 SIZE = 4096;
    std::vector<f32> func(SIZE * SIZE, 0.f);

    func[10000] = 1000.f;
    func[200000] = 10000000.f;
    func[300000] = 20000000.f;
    func[400000] = 300000000.f;
    func[500000] = 5000000000.f;

    std::mt19937 generator(73927932889);
    std::uniform_real_distribution<f32> distribution{};

    const auto dist = PiecewiseDist2D(func, SIZE, SIZE);

    constexpr i32 N = 32768;
    for (i32 i = 0; i < N; ++i) {
        const auto a = distribution(generator);
        const auto b = distribution(generator);
        const auto xi = vec2(a, b);

        const auto [xy, sample_pdf] = dist.sample(xi);
        const auto pdf = dist.pdf(xy);

        CHECK_THAT(pdf, Catch::Matchers::WithinRel(sample_pdf, 0.001f));
    }
}

TEST_CASE("piecewise2d pdf roundtrip image") {
    auto image = Image::make("../resources/test/abandoned_tank_farm_03_4k.exr");
    auto tex = ImageTexture(&image);

    std::vector<f32> sampling_grid(tex.width() * tex.height(), 0.f);

    const auto lambdas = SampledLambdas::new_sample_uniform(0.4f);

    spectral sum_rad = spectral::ZERO();

    for (int x = 0; x < tex.width(); ++x) {
        for (int y = 0; y < tex.height(); ++y) {
            const auto rgb = tex.fetch_rgb_texel(uvec2(x, y));
            const auto spec_illum = RgbSpectrumIlluminant::make(rgb, ColorSpace::sRGB);
            const auto spec = Spectrum(spec_illum);
            const auto rad = spec.eval(lambdas);

            sampling_grid[x + tex.width() * y] = (rgb.x + rgb.y + rgb.z) / 3.f;
            sum_rad += rad;
        }
    }

    auto dist = PiecewiseDist2D(sampling_grid, tex.width(), tex.height());

    std::mt19937 generator(73927932889);
    std::uniform_real_distribution<f32> distribution{};

    constexpr i32 N = 32768;
    for (i32 i = 0; i < N; ++i) {
        const auto a = distribution(generator);
        const auto b = distribution(generator);
        const auto xi = vec2(a, b);

        const auto [xy, sample_pdf] = dist.sample(xi);
        const auto pdf = dist.pdf(xy);

        CHECK_THAT(pdf, Catch::Matchers::WithinRel(sample_pdf, 0.001f));
    }
}

TEST_CASE("piecewise2d sampling") {
    // clang-format off
    const std::vector<f32> func = {
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
    };
    // clang-format on

    const auto dist = PiecewiseDist2D(func, 5, 5);

    std::mt19937 generator(73927932889);
    std::uniform_real_distribution<f32> distribution{};

    constexpr i32 N = 1024;
    for (i32 i = 0; i < N; ++i) {
        const auto a = distribution(generator);
        const auto b = distribution(generator);
        const auto xi = vec2(a, b);
        const auto [uv, pdf] = dist.sample(xi);

        REQUIRE_THAT(pdf, Catch::Matchers::WithinAbs(25.f, 0.0001f));

        REQUIRE_THAT(uv.x, Catch::Matchers::WithinAbs(0.7f, 0.1f));
        REQUIRE_THAT(uv.y, Catch::Matchers::WithinAbs(0.3f, 0.1f));
    }
}

TEST_CASE("piecewise1d pdf integrates to 1") {
    std::vector<f32> func = {1.f, 2.f, 4.f, 2.f, 1.f};
    const auto dist = PiecewiseDist1D(func);

    // TODO: replace with PCG or samplers ?
    std::mt19937 generator(73927932889);
    std::uniform_real_distribution<f32> distribution{};

    f32 sumpdf = 0.f;

    constexpr i32 N = 8192;

    for (i32 i = 0; i < N; ++i) {
        const auto xi = distribution(generator);
        sumpdf += dist.pdf(xi);
    }

    f32 integral = (1.f / N) * sumpdf;

    REQUIRE_THAT(integral, Catch::Matchers::WithinRel(1.f, 0.01f));
}

TEST_CASE("piecewise2d pdf integrates to 1") {
    // clang-format off
    const std::vector<f32> func = {
        1.f, 2.f, 4.f, 2.f,
        1.5f, 8.f, 4.f, 2.8f,
        17.f, 2.f, 3.f, 21.f,
        1.f, 3.2f, 1.4f, 5.f
    };
    // clang-format on

    const auto dist = PiecewiseDist2D(func, 4, 4);

    std::mt19937 generator(73927932889);
    std::uniform_real_distribution<f32> distribution{};

    f32 sumpdf = 0.f;

    constexpr i32 N = 8192;

    for (i32 i = 0; i < N; ++i) {
        const auto x_xi = distribution(generator);
        const auto y_xi = distribution(generator);
        sumpdf += dist.pdf(vec2(x_xi, y_xi));
    }

    f32 integral = (1.f / N) * sumpdf;

    REQUIRE_THAT(integral, Catch::Matchers::WithinRel(1.f, 0.01f));
}
