#include "pbrt_loader.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("loader camera", "[loader camera]") {
    auto input = std::string(R"(Camera "perspective" "float fov" [23] WorldBegin)");

    auto loader = PbrtLoader(input);
    Scene scene{};

    loader.load_scene(scene);

    REQUIRE(scene.attribs.camera.fov == 23.0f);
}

TEST_CASE("loader film", "[loader camera]") {
    auto input = std::string(R"(Film "rgb" "string filename" "simple.png"
     "integer xresolution" [400] "integer yresolution" [600] WorldBegin)");

    auto loader = PbrtLoader(input);
    Scene scene{};

    loader.load_scene(scene);

    REQUIRE(scene.attribs.film.filename == "simple.png");
    REQUIRE(scene.attribs.film.resx == 400);
    REQUIRE(scene.attribs.film.resy == 600);
}

// TODO: transform tests

TEST_CASE("loader_screenwide") {
    auto input = std::string(R"(
Scale -1 1 1
Film "rgb"
    "string filename" [ "foo.exr" ]
    "integer yresolution" [ 1400 ]
    "integer xresolution" [ 1000 ]
    "float iso" 150
    "string sensor" "canon_eos_5d_mkiv"

LookAt 0 5.5 24
    0 11 -10
    0 1 0
Camera "perspective"
    "float lensradius" [ 0.1 ]
    "float focaldistance" [ 17 ]
    "float fov" [ 47 ]
Sampler "halton"
    "integer pixelsamples" [ 512 ]
Integrator "volpath"
    "integer maxdepth" [ 100 ]
WorldBegin
)");

    auto loader = PbrtLoader(input);
    Scene scene{};

    loader.load_scene(scene);

    REQUIRE(scene.attribs.film.filename == "foo.exr");
    REQUIRE(scene.attribs.film.resy == 1400);
    REQUIRE(scene.attribs.film.resx == 1000);

    REQUIRE(scene.attribs.camera.fov == 47.f);
}

TEST_CASE("loader trianglemesh") {
    auto input =
        std::string(R"(WorldBegin Shape "trianglemesh"  "integer indices" [0 2 1 0 3 2 ]
    "point3 P" [550 0 0    0 0 0    0 0 560    550 0 560 ]
    "vector2 uv" [0 1  2 3  4 5  6 7]
    "normal N" [0 1 2  3 4 5  6 7 8  9 10 11])");

    auto loader = PbrtLoader(input);
    Scene scene{};

    loader.load_scene(scene);

    const auto &i = scene.geometry.meshes.meshes[0].indices;
    REQUIRE(i[0] == 0);
    REQUIRE(i[1] == 2);
    REQUIRE(i[2] == 1);
    REQUIRE(i[3] == 0);
    REQUIRE(i[4] == 3);
    REQUIRE(i[5] == 2);

    const auto &p = scene.geometry.meshes.meshes[0].pos;
    REQUIRE(p[0].x == 550);
    REQUIRE(p[0].y == 0);
    REQUIRE(p[0].z == 0);

    REQUIRE(p[1].x == 0);
    REQUIRE(p[1].y == 0);
    REQUIRE(p[1].z == 0);

    REQUIRE(p[2].x == 0);
    REQUIRE(p[2].y == 0);
    REQUIRE(p[2].z == 560);

    REQUIRE(p[3].x == 550);
    REQUIRE(p[3].y == 0);
    REQUIRE(p[3].z == 560);

    const auto &uv = scene.geometry.meshes.meshes[0].uvs;
    REQUIRE(uv[0].x == 0);
    REQUIRE(uv[0].y == 1);

    REQUIRE(uv[1].x == 2);
    REQUIRE(uv[1].y == 3);

    REQUIRE(uv[2].x == 4);
    REQUIRE(uv[2].y == 5);

    REQUIRE(uv[3].x == 6);
    REQUIRE(uv[3].y == 7);

    const auto &n = scene.geometry.meshes.meshes[0].normals;
    REQUIRE(n[0].x == 0);
    REQUIRE(n[0].y == 1);
    REQUIRE(n[0].z == 2);

    REQUIRE(n[1].x == 3);
    REQUIRE(n[1].y == 4);
    REQUIRE(n[1].z == 5);

    REQUIRE(n[2].x == 6);
    REQUIRE(n[2].y == 7);
    REQUIRE(n[2].z == 8);

    REQUIRE(n[3].x == 9);
    REQUIRE(n[3].y == 10);
    REQUIRE(n[3].z == 11);
}
