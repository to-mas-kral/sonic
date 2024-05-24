#include "discrete_dist.h"
#include "piecewise_dist.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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
