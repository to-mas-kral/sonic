
#include "algs.h"
#include "basic_types.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Binary interval search", "[binary_search_interval]") {
    std::vector<f32> vals{10.f, 20.f, 30.f, 40.f, 50.f};
    auto accessor = [&](size_t i) { return vals[i]; };

    REQUIRE(binary_search_interval(vals.size(), accessor, 10.f) == 0);
    REQUIRE(binary_search_interval(vals.size(), accessor, 50.f) == 3);

    REQUIRE(binary_search_interval(vals.size(), accessor, 15.f) == 0);
    REQUIRE(binary_search_interval(vals.size(), accessor, 25.f) == 1);
    REQUIRE(binary_search_interval(vals.size(), accessor, 35.f) == 2);
    REQUIRE(binary_search_interval(vals.size(), accessor, 45.f) == 3);
}

TEST_CASE("Binary interval search out of range", "[binary_search_interval]") {
    std::vector<f32> vals{10.f, 20.f, 30.f, 40.f};
    auto accessor = [&](size_t i) { return vals[i]; };

    REQUIRE(binary_search_interval(vals.size(), accessor, 0.f) == 0);
    REQUIRE(binary_search_interval(vals.size(), accessor, 50.f) == 2);
}

TEST_CASE("Binary interval search small", "[binary_search_interval]") {
    std::vector<f32> vals{10.f, 20.f};
    auto accessor = [&](size_t i) { return vals[i]; };

    REQUIRE(binary_search_interval(vals.size(), accessor, 0.f) == 0);
    REQUIRE(binary_search_interval(vals.size(), accessor, 30.f) == 0);
}
