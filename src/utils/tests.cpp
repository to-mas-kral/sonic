
#include "algs.h"
#include "basic_types.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Binary interval search", "[binary_search_interval]") {
    std::vector<f32> vals{10.F, 20.F, 30.F, 40.F, 50.F};
    auto accessor = [&](const size_t i) { return vals[i]; };

    REQUIRE(binary_search_interval(vals.size(), accessor, 10.F) == 0);
    REQUIRE(binary_search_interval(vals.size(), accessor, 50.F) == 3);

    REQUIRE(binary_search_interval(vals.size(), accessor, 15.F) == 0);
    REQUIRE(binary_search_interval(vals.size(), accessor, 25.F) == 1);
    REQUIRE(binary_search_interval(vals.size(), accessor, 35.F) == 2);
    REQUIRE(binary_search_interval(vals.size(), accessor, 45.F) == 3);

    REQUIRE(binary_search_interval(vals.size(), accessor, 20.F) == 1);
    REQUIRE(binary_search_interval(vals.size(), accessor, 30.F) == 2);
    REQUIRE(binary_search_interval(vals.size(), accessor, 40.F) == 3);
}

TEST_CASE("Binary interval search out of range", "[binary_search_interval]") {
    const std::vector<f32> vals{10.F, 20.F, 30.F, 40.F};
    auto accessor = [&](const size_t i) { return vals[i]; };

    REQUIRE(binary_search_interval(vals.size(), accessor, 0.F) == 0);
    REQUIRE(binary_search_interval(vals.size(), accessor, 50.F) == 2);
}

TEST_CASE("Binary interval search small", "[binary_search_interval]") {
    const std::vector<f32> vals{10.F, 20.F};
    auto accessor = [&](const size_t i) { return vals[i]; };

    REQUIRE(binary_search_interval(vals.size(), accessor, 0.F) == 0);
    REQUIRE(binary_search_interval(vals.size(), accessor, 30.F) == 0);
    REQUIRE(binary_search_interval(vals.size(), accessor, 10.F) == 0);
    REQUIRE(binary_search_interval(vals.size(), accessor, 20.F) == 0);
    REQUIRE(binary_search_interval(vals.size(), accessor, 15.F) == 0);
}
