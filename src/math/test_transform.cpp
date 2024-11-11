#include <catch2/catch_test_macros.hpp>

#include "transform.h"

TEST_CASE("transform translate", "[transform translate]") {
    const auto trans = mat4::from_translate(1.F, 2.F, 3.F);

    const auto res = trans.transform_point(point3(0.F, 0.F, 0.F));

    REQUIRE(res.x == 1.F);
    REQUIRE(res.y == 2.F);
    REQUIRE(res.z == 3.F);
}
