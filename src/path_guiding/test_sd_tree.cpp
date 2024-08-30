#include "../math/sampling.h"
#include "../scene/sphere_square_mapping.h"
#include "../utils/sampler.h"

#define TEST_PUBLIC
#include "sd_tree.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <matplot/matplot.h>

#include <vector>

TEST_CASE("quadtree node choose child") {
    QuadtreeNode node{};
    node.m_children = {0, 1, 2, 3};

    auto quadrant_half_0 = 0.125f;
    auto middle_0 = vec2(0.75f, 0.75f);
    const auto c_0 = node.choose_child(vec2(0.8f, 0.7f), middle_0, quadrant_half_0);

    REQUIRE(c_0 == 0);
    REQUIRE(middle_0.approx_eq(vec2(0.875f, 0.625f)));
    REQUIRE_THAT(quadrant_half_0, Catch::Matchers::WithinAbs(0.0625, 0.00001));

    auto quadrant_half_1 = 0.0625f;
    auto middle_1 = vec2(0.875f, 0.875f);
    const auto c_1 = node.choose_child(vec2(0.8f, 0.8f), middle_1, quadrant_half_1);

    REQUIRE(c_1 == 3);
    REQUIRE(middle_1.approx_eq(vec2(0.8125f, 0.8125f)));
    REQUIRE_THAT(quadrant_half_1, Catch::Matchers::WithinAbs(0.03125f, 0.00001f));

    auto quadrant_half_2 = 0.0625f;
    auto middle_2 = vec2(0.125f, 0.875f);
    const auto c_2 = node.choose_child(vec2(0.13f, 0.9f), middle_2, quadrant_half_2);

    REQUIRE(c_2 == 1);
    REQUIRE(middle_2.approx_eq(vec2(0.1875f, 0.9375f)));
    REQUIRE_THAT(quadrant_half_2, Catch::Matchers::WithinAbs(0.03125f, 0.00001f));
}

TEST_CASE("quadtree simple record") {
    Quadtree tree{};

    tree.nodes.resize(17);
    tree.nodes[0].m_children[0] = 1;
    tree.nodes[0].m_children[1] = 2;
    tree.nodes[0].m_children[2] = 3;
    tree.nodes[0].m_children[3] = 4;

    tree.nodes[4].m_children[0] = 5;
    tree.nodes[4].m_children[1] = 6;
    tree.nodes[4].m_children[2] = 7;
    tree.nodes[4].m_children[3] = 8;

    tree.nodes[5].m_children[0] = 9;
    tree.nodes[5].m_children[1] = 10;
    tree.nodes[5].m_children[2] = 11;
    tree.nodes[5].m_children[3] = 12;

    tree.nodes[10].m_children[0] = 13;
    tree.nodes[10].m_children[1] = 14;
    tree.nodes[10].m_children[2] = 15;
    tree.nodes[10].m_children[3] = 16;

    tree.record(spectral::ONE(), square_to_sphere(vec2(0.75f, 0.75f)));
    REQUIRE(tree.nodes[1].m_radiance == 0.f);
    REQUIRE(tree.nodes[2].m_radiance == 1.f);
    REQUIRE(tree.nodes[3].m_radiance == 0.f);
    REQUIRE(tree.nodes[4].m_radiance == 0.f);

    tree.reset_flux();

    tree.record(spectral::ONE(), square_to_sphere(vec2(0.75f, 0.25f)));
    REQUIRE(tree.nodes[1].m_radiance == 1.f);
    REQUIRE(tree.nodes[2].m_radiance == 0.f);
    REQUIRE(tree.nodes[3].m_radiance == 0.f);
    REQUIRE(tree.nodes[4].m_radiance == 0.f);

    tree.reset_flux();

    tree.record(spectral::ONE(), square_to_sphere(vec2(0.25f, 0.75f)));
    REQUIRE(tree.nodes[1].m_radiance == 0.f);
    REQUIRE(tree.nodes[2].m_radiance == 0.f);
    REQUIRE(tree.nodes[3].m_radiance == 1.f);
    REQUIRE(tree.nodes[4].m_radiance == 0.f);

    tree.reset_flux();

    tree.record(spectral::ONE(), square_to_sphere(vec2(0.375f, 0.375f)));
    REQUIRE(tree.nodes[1].m_radiance == 0.f);
    REQUIRE(tree.nodes[2].m_radiance == 0.f);
    REQUIRE(tree.nodes[3].m_radiance == 0.f);
    REQUIRE(tree.nodes[4].m_radiance == 1.f);
    REQUIRE(tree.nodes[5].m_radiance == 0.f);
    REQUIRE(tree.nodes[6].m_radiance == 1.f);
    REQUIRE(tree.nodes[7].m_radiance == 0.f);
    REQUIRE(tree.nodes[8].m_radiance == 0.f);

    tree.reset_flux();

    tree.record(spectral::ONE(), square_to_sphere(vec2(0.45f, 0.120f)));
    REQUIRE(tree.nodes[1].m_radiance == 0.f);
    REQUIRE(tree.nodes[2].m_radiance == 0.f);
    REQUIRE(tree.nodes[3].m_radiance == 0.f);
    REQUIRE(tree.nodes[4].m_radiance == 1.f);

    REQUIRE(tree.nodes[5].m_radiance == 1.f);
    REQUIRE(tree.nodes[6].m_radiance == 0.f);
    REQUIRE(tree.nodes[7].m_radiance == 0.f);
    REQUIRE(tree.nodes[8].m_radiance == 0.f);

    REQUIRE(tree.nodes[9].m_radiance == 1.f);
    REQUIRE(tree.nodes[10].m_radiance == 0.f);
    REQUIRE(tree.nodes[11].m_radiance == 0.f);
    REQUIRE(tree.nodes[12].m_radiance == 0.f);

    tree.reset_flux();

    tree.record(spectral::ONE(), square_to_sphere(vec2(0.41f, 0.19f)));
    REQUIRE(tree.nodes[1].m_radiance == 0.f);
    REQUIRE(tree.nodes[2].m_radiance == 0.f);
    REQUIRE(tree.nodes[3].m_radiance == 0.f);
    REQUIRE(tree.nodes[4].m_radiance == 1.f);

    REQUIRE(tree.nodes[5].m_radiance == 1.f);
    REQUIRE(tree.nodes[6].m_radiance == 0.f);
    REQUIRE(tree.nodes[7].m_radiance == 0.f);
    REQUIRE(tree.nodes[8].m_radiance == 0.f);

    REQUIRE(tree.nodes[9].m_radiance == 0.f);
    REQUIRE(tree.nodes[10].m_radiance == 1.f);
    REQUIRE(tree.nodes[11].m_radiance == 0.f);
    REQUIRE(tree.nodes[12].m_radiance == 0.f);

    REQUIRE(tree.nodes[13].m_radiance == 0.f);
    REQUIRE(tree.nodes[14].m_radiance == 0.f);
    REQUIRE(tree.nodes[15].m_radiance == 1.f);
    REQUIRE(tree.nodes[16].m_radiance == 0.f);
}

TEST_CASE("sample quadtree root") {
    auto tree = Quadtree();
    auto sampler = Sampler(uvec2(1, 1), uvec2(10, 10), 1, 10);

    u32 count_top = 0;
    u32 count_bottom = 0;
    u32 count_right = 0;
    u32 count_left = 0;

    constexpr i32 N = 1024;
    for (i32 i = 0; i < N; ++i) {
        auto pgsamle = tree.sample(sampler);
        auto xy = sphere_to_square(pgsamle.wi);

        REQUIRE_THAT(pgsamle.pdf,
                     Catch::Matchers::WithinAbs(1.f / (4.f * M_PIf), 0.0001));

        if (xy.x > 0.5f) {
            count_right++;
        } else {
            count_left++;
        }

        if (xy.y > 0.5f) {
            count_bottom++;
        } else {
            count_top++;
        }
    }

    constexpr u32 expected_count = N / 2;

    // TODO: actual chi-square test
    /*
    auto chi2_stat = 0.f;
    chi2_stat += sqr(count_top - expected_count) / expected_count;
    chi2_stat += sqr(count_bottom - expected_count) / expected_count;
    chi2_stat += sqr(count_right - expected_count) / expected_count;
    chi2_stat += sqr(count_left - expected_count) / expected_count;
    */

    REQUIRE(count_top > expected_count - 100);
    REQUIRE(count_bottom > expected_count - 100);
    REQUIRE(count_right > expected_count - 100);
    REQUIRE(count_left > expected_count - 100);
}

TEST_CASE("quadtree refine and sample level 1") {
    auto tree = Quadtree();
    auto sampler = Sampler(uvec2(1, 1), uvec2(10, 10), 1, 10);

    for (int i = 0; i < 128; ++i) {
        const auto xi_x = sampler.sample();
        const auto xi_y = sampler.sample();

        const auto wi = sample_uniform_sphere(vec2(xi_x, xi_y));

        tree.record(spectral::ONE(), wi);
    }

    tree.refine();

    u32 count_top = 0;
    u32 count_bottom = 0;
    u32 count_right = 0;
    u32 count_left = 0;

    constexpr i32 N = 1024;
    for (i32 i = 0; i < N; ++i) {
        auto pgsamle = tree.sample(sampler);
        auto xy = sphere_to_square(pgsamle.wi);

        REQUIRE_THAT(pgsamle.pdf,
                     Catch::Matchers::WithinAbs(1.f / (4.f * M_PIf), 0.0001));

        if (xy.x > 0.5f) {
            count_right++;
        } else {
            count_left++;
        }

        if (xy.y > 0.5f) {
            count_bottom++;
        } else {
            count_top++;
        }
    }

    constexpr u32 expected_count = N / 2;

    // TODO: actual chi-square test
    /*
    auto chi2_stat = 0.f;
    chi2_stat += sqr(count_top - expected_count) / expected_count;
    chi2_stat += sqr(count_bottom - expected_count) / expected_count;
    chi2_stat += sqr(count_right - expected_count) / expected_count;
    chi2_stat += sqr(count_left - expected_count) / expected_count;
    */

    REQUIRE(count_top > expected_count - 100);
    REQUIRE(count_bottom > expected_count - 100);
    REQUIRE(count_right > expected_count - 100);
    REQUIRE(count_left > expected_count - 100);
}

TEST_CASE("quadtree refine multiple and sample level 1") {
    auto tree = Quadtree();
    auto sampler = Sampler(uvec2(1, 1), uvec2(10, 10), 1, 10);

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 1024; ++j) {
            f32 xi_x = sampler.sample();
            f32 xi_y = sampler.sample();

            const auto wi = sample_uniform_sphere(vec2(xi_x, xi_y));

            tree.record(spectral::ONE(), wi);
        }

        tree.refine();

        if (i != 7) {
            tree.reset_flux();
        }
    }

    std::vector<f64> sampled_x{};
    std::vector<f64> sampled_y{};

    u32 count_top = 0;
    u32 count_bottom = 0;
    u32 count_right = 0;
    u32 count_left = 0;

    constexpr i32 N = 1024;
    for (i32 i = 0; i < N; ++i) {
        auto pgsamle = tree.sample(sampler);
        auto xy = sphere_to_square(pgsamle.wi);

        if (xy.x > 0.5f) {
            count_right++;
        } else {
            count_left++;
        }

        if (xy.y > 0.5f) {
            count_bottom++;
        } else {
            count_top++;
        }

        sampled_x.push_back(xy.x);
        sampled_y.push_back(xy.y);
    }

    constexpr u32 expected_count = N / 2;

    // TODO: actual chi-square test
    /*
    auto chi2_stat = 0.f;
    chi2_stat += sqr(count_top - expected_count) / expected_count;
    chi2_stat += sqr(count_bottom - expected_count) / expected_count;
    chi2_stat += sqr(count_right - expected_count) / expected_count;
    chi2_stat += sqr(count_left - expected_count) / expected_count;
    */

    REQUIRE(count_top > expected_count - 100);
    REQUIRE(count_bottom > expected_count - 100);
    REQUIRE(count_right > expected_count - 100);
    REQUIRE(count_left > expected_count - 100);
}

TEST_CASE("quadtree refine uneven and sample level 1") {
    auto tree = Quadtree();
    auto sampler = Sampler(uvec2(1, 1), uvec2(10, 10), 1, 10);

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 1024; ++j) {
            const auto xi = sampler.sample();

            f32 xi_x = sampler.sample();
            f32 xi_y;

            if (xi < 0.25f) {
                xi_y = sampler.sample() / 2.f;
            } else {
                xi_y = 0.5f + sampler.sample() / 2.f;
            }

            const auto wi = square_to_sphere(vec2(xi_x, xi_y));

            tree.record(spectral::ONE(), wi);
        }

        tree.refine();

        if (i != 7) {
            tree.reset_flux();
        }
    }

    std::vector<f64> sampled_x{};
    std::vector<f64> sampled_y{};

    u32 count_top = 0;
    u32 count_bottom = 0;
    u32 count_right = 0;
    u32 count_left = 0;

    constexpr i32 N = 1024;
    for (i32 i = 0; i < N; ++i) {
        auto pgsamle = tree.sample(sampler);
        auto xy = sphere_to_square(pgsamle.wi);

        if (xy.x > 0.5f) {
            count_right++;
        } else {
            count_left++;
        }

        if (xy.y > 0.5f) {
            count_bottom++;
        } else {
            count_top++;
        }

        sampled_x.push_back(xy.x);
        sampled_y.push_back(xy.y);
    }

    constexpr u32 expected_count = N / 2;
    constexpr u32 expected_top = 256;
    constexpr u32 expected_bottom = 768;

    // TODO: actual chi-square test
    /*
    auto chi2_stat = 0.f;
    chi2_stat += sqr(count_top - expected_count) / expected_count;
    chi2_stat += sqr(count_bottom - expected_count) / expected_count;
    chi2_stat += sqr(count_right - expected_count) / expected_count;
    chi2_stat += sqr(count_left - expected_count) / expected_count;
    */

    REQUIRE(count_top > expected_top - 100);
    REQUIRE(count_bottom > expected_bottom - 100);
    REQUIRE(count_right > expected_count - 100);
    REQUIRE(count_left > expected_count - 100);
}
