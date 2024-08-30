
#include "../utils/basic_types.h"
#include "morton.h"

#include "catch2/catch_test_macros.hpp"
#include <matplot/matplot.h>

#include <limits>

TEST_CASE("morton encoding") {
    CHECK(morton_encode(uvec2(0, 0)) == 0);
    CHECK(morton_encode(uvec2(1, 0)) == 1);
    CHECK(morton_encode(uvec2(0, 1)) == 2);
    CHECK(morton_encode(uvec2(1, 1)) == 3);
    CHECK(morton_encode(uvec2(2, 0)) == 4);
    CHECK(morton_encode(uvec2(3, 0)) == 5);
    CHECK(morton_encode(uvec2(2, 1)) == 6);
    CHECK(morton_encode(uvec2(3, 1)) == 7);
}

TEST_CASE("morton decoding") {
    CHECK(morton_decode(0) == uvec2(0, 0));
    CHECK(morton_decode(1) == uvec2(1, 0));
    CHECK(morton_decode(2) == uvec2(0, 1));
    CHECK(morton_decode(3) == uvec2(1, 1));
    CHECK(morton_decode(4) == uvec2(2, 0));
    CHECK(morton_decode(5) == uvec2(3, 0));
    CHECK(morton_decode(6) == uvec2(2, 1));
    CHECK(morton_decode(7) == uvec2(3, 1));
}

// This works but is too slow...
/*
TEST_CASE("morton roundtrip") {
    for (u64 i = 0; i < std::numeric_limits<u64>::max(); ++i) {
        const auto decoded = morton_decode(i);
        const auto encoded = morton_encode(decoded);
        REQUIRE(i == encoded);
    }
}
*/
