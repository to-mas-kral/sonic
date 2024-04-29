#include <catch2/catch_test_macros.hpp>

#include "transform.h"

TEST_CASE("transform translate", "[transform translate]") {
    const auto trans = mat4::from_translate(1.f, 2.f, 3.f);

    const auto res = trans.transform_point(point3(0.f, 0.f, 0.f));

    REQUIRE(res.x == 1.f);
    REQUIRE(res.y == 2.f);
    REQUIRE(res.z == 3.f);
}
