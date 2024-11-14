#include "../scene/envmap.h"
#include "aabb.h"
#include "discrete_dist.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>


TEST_CASE("AABB split halfs") {
    auto low = vec3(-0.5f, -1.f, 0.3f);
    auto high = vec3(1.2f, 0.6f, 2.f);
    auto bounds = AABB(low, high);

    auto right_x = bounds.right_half(Axis::X);

    REQUIRE(right_x.low.approx_eq(vec3(0.35f, -1.f, 0.3f)));
    REQUIRE(right_x.high.approx_eq(high));

    auto left_x = bounds.left_half(Axis::X);

    REQUIRE(left_x.low.approx_eq(low));
    REQUIRE(left_x.high.approx_eq(vec3(0.35f, 0.6f, 2.f)));

    auto right_y = bounds.right_half(Axis::Y);

    REQUIRE(right_y.low.approx_eq(vec3(-0.5f, -0.2f, 0.3f)));
    REQUIRE(right_y.high.approx_eq(high));

    auto left_y = bounds.left_half(Axis::Y);

    REQUIRE(left_y.low.approx_eq(low));
    REQUIRE(left_y.high.approx_eq(vec3(1.2f, -0.2f, 2.f)));

    auto right_z = bounds.right_half(Axis::Z);

    REQUIRE(right_z.low.approx_eq(vec3(-0.5f, -1.f, 1.15f)));
    REQUIRE(right_z.high.approx_eq(high));

    auto left_z = bounds.left_half(Axis::Z);

    REQUIRE(left_z.low.approx_eq(low));
    REQUIRE(left_z.high.approx_eq(vec3(1.2f, 0.6f, 1.15f)));
}

TEST_CASE("AABB contains") {
    auto low = vec3(-0.5f, -1.f, 0.3f);
    auto high = vec3(1.2f, 0.6f, 2.f);
    auto bounds = AABB(low, high);

    REQUIRE(bounds.contains(point3(0.f, 0.f, 0.4f)));
    REQUIRE(bounds.contains(point3(-0.4f, -0.9f, 0.4f)));

    REQUIRE(!bounds.contains(point3(1.21f, 0.6f, 1.9f)));
    REQUIRE(!bounds.contains(point3(0.f, 0.f, 0.f)));
    REQUIRE(!bounds.contains(point3(1.f, 1.f, 1.f)));
    REQUIRE(!bounds.contains(point3(-1.f, -1.f, -1.f)));
}

TEST_CASE("AABB right_of_split_axis") {
    auto low = vec3(-0.5f, -1.f, 0.3f);
    auto high = vec3(1.2f, 0.6f, 2.f);
    auto bounds = AABB(low, high);

    REQUIRE(!bounds.right_of_split_axis(point3(0.2f, -1.f, 0.3f), Axis::X));
    REQUIRE(bounds.right_of_split_axis(point3(0.45f, -1.f, 0.3f), Axis::X));

    REQUIRE(!bounds.right_of_split_axis(point3(0.4f, -0.3f, 0.3f), Axis::Y));
    REQUIRE(bounds.right_of_split_axis(point3(0.4f, 0.2f, 0.3f), Axis::Y));

    REQUIRE(!bounds.right_of_split_axis(point3(-0.4f, -0.2f, 1.1f), Axis::Z));
    REQUIRE(bounds.right_of_split_axis(point3(0.4f, -0.2f, 1.2f), Axis::Z));
}
