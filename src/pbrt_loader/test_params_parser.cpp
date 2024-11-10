#define TEST_PUBLIC
#include "pbrt_loader.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("params simple", "[params simple]") {
    auto input = std::string(R"("checks" "spectrum" "checkerboard")");

    auto loader = PbrtLoader(input);

    const auto params = loader.parse_param_list().params;

    REQUIRE(params[0].type == ParamType::Simple);
    REQUIRE(params[0].name == "checks");

    REQUIRE(params[1].type == ParamType::Simple);
    REQUIRE(params[1].name == "spectrum");

    REQUIRE(params[2].type == ParamType::Simple);
    REQUIRE(params[2].name == "checkerboard");
}

TEST_CASE("params single", "[params single]") {
    auto input = std::string(R"("float a" [1] "integer b" 1)");

    auto loader = PbrtLoader(input);

    const auto params = loader.parse_param_list().params;

    REQUIRE(params[0].type == ParamType::Single);
    REQUIRE(params[0].name == "a");
    REQUIRE(params[0].value_type == ValueType::Float);
    REQUIRE(std::get<f32>(params[0].inner) == 1.0f);

    REQUIRE(params[1].type == ParamType::Single);
    REQUIRE(params[1].name == "b");
    REQUIRE(params[1].value_type == ValueType::Int);
    REQUIRE(std::get<i32>(params[1].inner) == 1);
}

TEST_CASE("params list", "[params list]") {
    auto input = std::string(R"("integer b" [1 2 3 4 5] "float a" [1 2 3]
"integer c" [1 2 3 4 5])");

    auto loader = PbrtLoader(input);

    const auto params = loader.parse_param_list().params;

    REQUIRE(params[0].type == ParamType::List);
    REQUIRE(params[0].name == "b");
    REQUIRE(params[0].value_type == ValueType::Int);
    REQUIRE(std::get<std::vector<i32>>(params[0].inner) ==
            std::vector<i32>{1, 2, 3, 4, 5});

    REQUIRE(params[1].type == ParamType::List);
    REQUIRE(params[1].name == "a");
    REQUIRE(params[1].value_type == ValueType::Float);
    REQUIRE(std::get<std::vector<f32>>(params[1].inner) == std::vector{1.0f, 2.0f, 3.0f});

    REQUIRE(params[2].type == ParamType::List);
    REQUIRE(params[2].name == "c");
    REQUIRE(params[2].value_type == ValueType::Int);
    REQUIRE(std::get<std::vector<i32>>(params[2].inner) ==
            std::vector<i32>{1, 2, 3, 4, 5});
}

TEST_CASE("params various", "[params various]") {
    auto input = std::string(R"("distant" "point3 from" [ -30 40  100 ]
    "float scale" 1.5)");

    auto loader = PbrtLoader(input);

    const auto params = loader.parse_param_list().params;

    REQUIRE(params[0].type == ParamType::Simple);
    REQUIRE(params[0].name == "distant");

    REQUIRE(params[1].type == ParamType::Single);
    REQUIRE(params[1].name == "from");
    REQUIRE(params[1].value_type == ValueType::Point3);

    const auto p = std::get<point3>(params[1].inner);
    REQUIRE(p.x == -30.0f);
    REQUIRE(p.y == 40.0f);
    REQUIRE(p.z == 100.0f);

    REQUIRE(params[2].type == ParamType::Single);
    REQUIRE(params[2].name == "scale");
    REQUIRE(params[2].value_type == ValueType::Float);
    REQUIRE(std::get<f32>(params[2].inner) == 1.5f);
}

TEST_CASE("params duplicated param name") {
    const auto input = std::string(R"("arc_rosetta_red-kd" "spectrum" "scale"
    "float scale" [ 0.620968 ])");

    auto loader = PbrtLoader(input);
    auto params = loader.parse_param_list();

    REQUIRE_NOTHROW(std::get<f32>(params.get_required("scale", ValueType::Float).inner));
}
