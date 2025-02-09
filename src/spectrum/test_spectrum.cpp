#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "spectrum.h"

TEST_CASE("dense spectrum") {
    const auto spectrum = Spectrum(CIE_X);

    REQUIRE_THAT(spectrum.eval_single(LAMBDA_MIN),
                 Catch::Matchers::WithinRel(CIE_X_RAW.front(), 0.0001f));

    REQUIRE_THAT(spectrum.eval_single(LAMBDA_MIN + 8),
                 Catch::Matchers::WithinRel(CIE_X_RAW[8], 0.0001f));

    REQUIRE_THAT(spectrum.eval_single(LAMBDA_MAX),
                 Catch::Matchers::WithinRel(CIE_X_RAW.back(), 0.0001f));
}

TEST_CASE("piecewise spectrum") {
    // clang-format off
    static constexpr auto PIECEWISE_SPECTRUM_RAW = std::array{
        400.F, 1.F,
        450.F, 2.F,
        500.F, 3.F,
    };
    // clang-format on

    constexpr auto piecewise_spectrum = PiecewiseSpectrum(
        std::span(PIECEWISE_SPECTRUM_RAW.data(), PIECEWISE_SPECTRUM_RAW.size()));
    const auto spectrum = Spectrum(piecewise_spectrum);

    REQUIRE_THAT(spectrum.eval_single(400.F), Catch::Matchers::WithinRel(1.F, 0.0001f));
    REQUIRE_THAT(spectrum.eval_single(450.F), Catch::Matchers::WithinRel(2.F, 0.0001f));
    REQUIRE_THAT(spectrum.eval_single(500.F), Catch::Matchers::WithinRel(3.F, 0.0001f));
    REQUIRE_THAT(spectrum.eval_single(360.F), Catch::Matchers::WithinRel(0.F, 0.f));
    REQUIRE_THAT(spectrum.eval_single(830.F), Catch::Matchers::WithinRel(0.F, 0.f));

    REQUIRE_THAT(spectrum.eval_single(425.F), Catch::Matchers::WithinRel(1.5F, 0.0001f));
    REQUIRE_THAT(spectrum.eval_single(475.F), Catch::Matchers::WithinRel(2.5F, 0.0001f));
}
