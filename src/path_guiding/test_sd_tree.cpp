#include "../math/samplers/sampler.h"
#include "../math/sampling.h"
#include "../scene/sphere_square_mapping.h"

#define TEST_PUBLIC
#include "sd_tree.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>

TEST_CASE("quadtree node choose child") {
    QuadtreeNode node{};
    node.m_children = {0, 1, 2, 3};

    auto quadrant_half_0 = 0.125F;
    auto middle_0 = vec2(0.75F, 0.75F);
    const auto c_0 = node.choose_child(vec2(0.8F, 0.7F), middle_0, quadrant_half_0);

    REQUIRE(c_0.parent_index == 0);
    REQUIRE(middle_0.approx_eq(vec2(0.875F, 0.625F)));
    REQUIRE_THAT(quadrant_half_0, Catch::Matchers::WithinAbs(0.0625F, 0.00001F));

    auto quadrant_half_1 = 0.0625F;
    auto middle_1 = vec2(0.875F, 0.875F);
    const auto c_1 = node.choose_child(vec2(0.8F, 0.8F), middle_1, quadrant_half_1);

    REQUIRE(c_1.parent_index == 3);
    REQUIRE(middle_1.approx_eq(vec2(0.8125F, 0.8125F)));
    REQUIRE_THAT(quadrant_half_1, Catch::Matchers::WithinAbs(0.03125F, 0.00001F));

    auto quadrant_half_2 = 0.0625F;
    auto middle_2 = vec2(0.125F, 0.875F);
    const auto c_2 = node.choose_child(vec2(0.13F, 0.9F), middle_2, quadrant_half_2);

    REQUIRE(c_2.parent_index == 1);
    REQUIRE(middle_2.approx_eq(vec2(0.1875F, 0.9375F)));
    REQUIRE_THAT(quadrant_half_2, Catch::Matchers::WithinAbs(0.03125F, 0.00001F));
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

    tree.record(spectral::ONE(), square_to_sphere(vec2(0.75F, 0.75F)));
    REQUIRE(tree.nodes[1].m_radiance == 0.F);
    REQUIRE(tree.nodes[2].m_radiance == 1.F);
    REQUIRE(tree.nodes[3].m_radiance == 0.F);
    REQUIRE(tree.nodes[4].m_radiance == 0.F);

    tree.reset_flux();

    tree.record(spectral::ONE(), square_to_sphere(vec2(0.75F, 0.25F)));
    REQUIRE(tree.nodes[1].m_radiance == 1.F);
    REQUIRE(tree.nodes[2].m_radiance == 0.F);
    REQUIRE(tree.nodes[3].m_radiance == 0.F);
    REQUIRE(tree.nodes[4].m_radiance == 0.F);

    tree.reset_flux();

    tree.record(spectral::ONE(), square_to_sphere(vec2(0.25F, 0.75F)));
    REQUIRE(tree.nodes[1].m_radiance == 0.F);
    REQUIRE(tree.nodes[2].m_radiance == 0.F);
    REQUIRE(tree.nodes[3].m_radiance == 1.F);
    REQUIRE(tree.nodes[4].m_radiance == 0.F);

    tree.reset_flux();

    tree.record(spectral::ONE(), square_to_sphere(vec2(0.375F, 0.375F)));
    REQUIRE(tree.nodes[1].m_radiance == 0.F);
    REQUIRE(tree.nodes[2].m_radiance == 0.F);
    REQUIRE(tree.nodes[3].m_radiance == 0.F);
    REQUIRE(tree.nodes[4].m_radiance == 1.F);
    REQUIRE(tree.nodes[5].m_radiance == 0.F);
    REQUIRE(tree.nodes[6].m_radiance == 1.F);
    REQUIRE(tree.nodes[7].m_radiance == 0.F);
    REQUIRE(tree.nodes[8].m_radiance == 0.F);

    tree.reset_flux();

    tree.record(spectral::ONE(), square_to_sphere(vec2(0.45F, 0.120F)));
    REQUIRE(tree.nodes[1].m_radiance == 0.F);
    REQUIRE(tree.nodes[2].m_radiance == 0.F);
    REQUIRE(tree.nodes[3].m_radiance == 0.F);
    REQUIRE(tree.nodes[4].m_radiance == 1.F);

    REQUIRE(tree.nodes[5].m_radiance == 1.F);
    REQUIRE(tree.nodes[6].m_radiance == 0.F);
    REQUIRE(tree.nodes[7].m_radiance == 0.F);
    REQUIRE(tree.nodes[8].m_radiance == 0.F);

    REQUIRE(tree.nodes[9].m_radiance == 1.F);
    REQUIRE(tree.nodes[10].m_radiance == 0.F);
    REQUIRE(tree.nodes[11].m_radiance == 0.F);
    REQUIRE(tree.nodes[12].m_radiance == 0.F);

    tree.reset_flux();

    tree.record(spectral::ONE(), square_to_sphere(vec2(0.41F, 0.19F)));
    REQUIRE(tree.nodes[1].m_radiance == 0.F);
    REQUIRE(tree.nodes[2].m_radiance == 0.F);
    REQUIRE(tree.nodes[3].m_radiance == 0.F);
    REQUIRE(tree.nodes[4].m_radiance == 1.F);

    REQUIRE(tree.nodes[5].m_radiance == 1.F);
    REQUIRE(tree.nodes[6].m_radiance == 0.F);
    REQUIRE(tree.nodes[7].m_radiance == 0.F);
    REQUIRE(tree.nodes[8].m_radiance == 0.F);

    REQUIRE(tree.nodes[9].m_radiance == 0.F);
    REQUIRE(tree.nodes[10].m_radiance == 1.F);
    REQUIRE(tree.nodes[11].m_radiance == 0.F);
    REQUIRE(tree.nodes[12].m_radiance == 0.F);

    REQUIRE(tree.nodes[13].m_radiance == 0.F);
    REQUIRE(tree.nodes[14].m_radiance == 0.F);
    REQUIRE(tree.nodes[15].m_radiance == 1.F);
    REQUIRE(tree.nodes[16].m_radiance == 0.F);
}

// TODO: chi-square tests
// TODO: refactor parts into common functions

TEST_CASE("sample quadtree root") {
    auto tree = Quadtree();
    auto sampler = Sampler(uvec2(1, 1), uvec2(10, 10), 10);

    u32 count_top = 0;
    u32 count_bottom = 0;
    u32 count_right = 0;
    u32 count_left = 0;

    constexpr i32 N = 1024;
    for (i32 i = 0; i < N; ++i) {
        auto pgsamle = tree.sample(sampler);
        auto xy = sphere_to_square(pgsamle.wi);

        REQUIRE_THAT(pgsamle.pdf,
                     Catch::Matchers::WithinAbs(1.F / (4.f * M_PIf), 0.0001));

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

    REQUIRE(count_top > expected_count - 100);
    REQUIRE(count_bottom > expected_count - 100);
    REQUIRE(count_right > expected_count - 100);
    REQUIRE(count_left > expected_count - 100);
}

TEST_CASE("quadtree refine and sample level 1") {
    auto tree = Quadtree();
    auto sampler = Sampler(uvec2(1, 1), uvec2(10, 10), 10);

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
                     Catch::Matchers::WithinAbs(1.F / (4.F * M_PIf), 0.0001));

        if (xy.x > 0.5F) {
            count_right++;
        } else {
            count_left++;
        }

        if (xy.y > 0.5F) {
            count_bottom++;
        } else {
            count_top++;
        }
    }

    constexpr u32 expected_count = N / 2;

    REQUIRE(count_top > expected_count - 100);
    REQUIRE(count_bottom > expected_count - 100);
    REQUIRE(count_right > expected_count - 100);
    REQUIRE(count_left > expected_count - 100);
}

TEST_CASE("quadtree refine multiple and sample level 1") {
    auto tree = Quadtree();
    auto sampler = Sampler(uvec2(1, 1), uvec2(10, 10), 10);

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

        if (xy.x > 0.5F) {
            count_right++;
        } else {
            count_left++;
        }

        if (xy.y > 0.5F) {
            count_bottom++;
        } else {
            count_top++;
        }

        sampled_x.push_back(xy.x);
        sampled_y.push_back(xy.y);
    }

    constexpr u32 expected_count = N / 2;

    REQUIRE(count_top > expected_count - 100);
    REQUIRE(count_bottom > expected_count - 100);
    REQUIRE(count_right > expected_count - 100);
    REQUIRE(count_left > expected_count - 100);
}

TEST_CASE("quadtree refine uneven and sample level 1") {
    auto tree = Quadtree();
    auto sampler = Sampler(uvec2(1, 1), uvec2(10, 10), 10);

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 1024; ++j) {
            const auto xi = sampler.sample();

            f32 xi_x = sampler.sample();
            f32 xi_y;

            if (xi < 0.25F) {
                xi_y = sampler.sample() / 2.F;
            } else {
                xi_y = 0.5F + sampler.sample() / 2.F;
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

        if (xy.x > 0.5F) {
            count_right++;
        } else {
            count_left++;
        }

        if (xy.y > 0.5F) {
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

    REQUIRE(count_top > expected_top - 100);
    REQUIRE(count_bottom > expected_bottom - 100);
    REQUIRE(count_right > expected_count - 100);
    REQUIRE(count_left > expected_count - 100);
}

TEST_CASE("quadtree sampling pdf integrates to 1") {
    auto tree = Quadtree();
    auto sampler = Sampler(uvec2(1, 1), uvec2(10, 10), 10);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 1024U << i; ++j) {
            const f32 xi_x = sampler.sample();
            const f32 xi_y = sampler.sample();

            const auto wi = square_to_sphere(vec2(xi_x, xi_y));
            tree.record(spectral::make_constant(1.F), wi);
        }

        if (i != 2) {
            tree.refine();
            tree.reset_flux();
        }
    }

    tree.refine();

    auto dimsampler = DimensionSampler();
    constexpr i32 ITERS = 2048;
    f64 pdf_sum = 0.;
    for (int i = 0; i < ITERS; ++i) {
        const auto sample = tree.pdf(square_to_sphere(dimsampler.sample2()));
        pdf_sum += sample;
    }

    const auto integral = 4. * M_PI * pdf_sum / static_cast<f64>(ITERS);
    REQUIRE_THAT(integral, Catch::Matchers::WithinRelMatcher(1., 0.01));
}

TEST_CASE("quadtree refine and split 1 level") {
    Quadtree tree{};

    tree.nodes.resize(5);
    tree.nodes[0].m_children[0] = 1;
    tree.nodes[0].m_children[1] = 2;
    tree.nodes[0].m_children[2] = 3;
    tree.nodes[0].m_children[3] = 4;

    tree.nodes[0].record_radiance(spectral::make_constant(1.0));
    tree.nodes[1].record_radiance(spectral::make_constant(0.2));
    tree.nodes[2].record_radiance(spectral::make_constant(0.2));
    tree.nodes[3].record_radiance(spectral::make_constant(0.2));
    tree.nodes[4].record_radiance(spectral::make_constant(0.4));

    tree.refine(0.25);

    REQUIRE(tree.nodes[1].is_leaf());
    REQUIRE(tree.nodes[2].is_leaf());
    REQUIRE(tree.nodes[3].is_leaf());
    REQUIRE(!tree.nodes[4].is_leaf());

    REQUIRE(tree.nodes[4].m_children[0] == 5);
    REQUIRE(tree.nodes[4].m_children[1] == 6);
    REQUIRE(tree.nodes[4].m_children[2] == 7);
    REQUIRE(tree.nodes[4].m_children[3] == 8);

    REQUIRE_THAT(tree.nodes[0].m_radiance, Catch::Matchers::WithinRel(1.0, 0.01));
    REQUIRE_THAT(tree.nodes[1].m_radiance, Catch::Matchers::WithinRel(0.2, 0.01));
    REQUIRE_THAT(tree.nodes[2].m_radiance, Catch::Matchers::WithinRel(0.2, 0.01));
    REQUIRE_THAT(tree.nodes[3].m_radiance, Catch::Matchers::WithinRel(0.2, 0.01));
    REQUIRE_THAT(tree.nodes[4].m_radiance, Catch::Matchers::WithinRel(0.4, 0.01));

    REQUIRE_THAT(tree.nodes[5].m_radiance, Catch::Matchers::WithinRel(0.1, 0.01));
    REQUIRE_THAT(tree.nodes[6].m_radiance, Catch::Matchers::WithinRel(0.1, 0.01));
    REQUIRE_THAT(tree.nodes[7].m_radiance, Catch::Matchers::WithinRel(0.1, 0.01));
    REQUIRE_THAT(tree.nodes[8].m_radiance, Catch::Matchers::WithinRel(0.1, 0.01));
}

TEST_CASE("quadtree refine split and prune") {
    Quadtree tree{};

    tree.nodes.resize(50);
    tree.nodes[0].m_children[0] = 1;
    tree.nodes[0].m_children[1] = 2;
    tree.nodes[0].m_children[2] = 3;
    tree.nodes[0].m_children[3] = 4;

    tree.nodes[1].m_children[0] = 5;
    tree.nodes[1].m_children[1] = 6;
    tree.nodes[1].m_children[2] = 7;
    tree.nodes[1].m_children[3] = 8;

    tree.nodes[0].record_radiance(spectral::make_constant(8.1));
    tree.nodes[1].record_radiance(spectral::make_constant(0.05));
    tree.nodes[2].record_radiance(spectral::make_constant(0.025));
    tree.nodes[3].record_radiance(spectral::make_constant(0.025));
    tree.nodes[4].record_radiance(spectral::make_constant(8.0));

    tree.refine(0.2);

    REQUIRE(tree.nodes[1].is_leaf());
    REQUIRE(tree.nodes[2].is_leaf());
    REQUIRE(tree.nodes[3].is_leaf());
    REQUIRE(!tree.nodes[4].is_leaf());

    REQUIRE(tree.nodes[4].m_children[0] == 5);
    REQUIRE(tree.nodes[4].m_children[1] == 6);
    REQUIRE(tree.nodes[4].m_children[2] == 7);
    REQUIRE(tree.nodes[4].m_children[3] == 8);

    REQUIRE_THAT(tree.nodes[0].m_radiance, Catch::Matchers::WithinRel(8.1, 0.01));
    REQUIRE_THAT(tree.nodes[1].m_radiance, Catch::Matchers::WithinRel(0.05, 0.01));
    REQUIRE_THAT(tree.nodes[2].m_radiance, Catch::Matchers::WithinRel(0.025, 0.01));
    REQUIRE_THAT(tree.nodes[3].m_radiance, Catch::Matchers::WithinRel(0.025, 0.01));
    REQUIRE_THAT(tree.nodes[4].m_radiance, Catch::Matchers::WithinRel(8.0, 0.01));

    REQUIRE_THAT(tree.nodes[5].m_radiance, Catch::Matchers::WithinRel(2.0, 0.01));
    REQUIRE_THAT(tree.nodes[6].m_radiance, Catch::Matchers::WithinRel(2.0, 0.01));
    REQUIRE_THAT(tree.nodes[7].m_radiance, Catch::Matchers::WithinRel(2.0, 0.01));
    REQUIRE_THAT(tree.nodes[8].m_radiance, Catch::Matchers::WithinRel(2.0, 0.01));

    REQUIRE(tree.nodes[5].m_children[0] == 9);
    REQUIRE(tree.nodes[5].m_children[1] == 10);
    REQUIRE(tree.nodes[5].m_children[2] == 11);
    REQUIRE(tree.nodes[5].m_children[3] == 12);

    REQUIRE(tree.nodes[6].m_children[0] == 13);
    REQUIRE(tree.nodes[6].m_children[1] == 14);
    REQUIRE(tree.nodes[6].m_children[2] == 15);
    REQUIRE(tree.nodes[6].m_children[3] == 16);

    REQUIRE(tree.nodes[7].m_children[0] == 17);
    REQUIRE(tree.nodes[7].m_children[1] == 18);
    REQUIRE(tree.nodes[7].m_children[2] == 19);
    REQUIRE(tree.nodes[7].m_children[3] == 20);

    REQUIRE(tree.nodes[8].m_children[0] == 21);
    REQUIRE(tree.nodes[8].m_children[1] == 22);
    REQUIRE(tree.nodes[8].m_children[2] == 23);
    REQUIRE(tree.nodes[8].m_children[3] == 24);

    for (int i = 9; i < 25; ++i) {
        REQUIRE(tree.nodes[i].m_radiance == 0.5);
    }
}

TEST_CASE("spatial tree splitting") {
    SDTree tree(AABB(vec3(-1.F, -1.F, -1.F), vec3(1.F, 1.F, 1.F)));

    tree.record_bulk(point3(0.F), spectral::ZERO(), norm_vec3(), 100000,
                     SampledLambdas::new_mock(), MaterialId(0));
    tree.refine(0);

    REQUIRE(tree.nodes.size() == 3);

    tree.record_bulk(point3(-0.5F, 0.F, 0.F), spectral::ZERO(), norm_vec3(), 10,
                     SampledLambdas::new_mock(), MaterialId(0));
    tree.record_bulk(point3(0.5F, 0.F, 0.F), spectral::ZERO(), norm_vec3(), 100000,
                     SampledLambdas::new_mock(), MaterialId(0));

    REQUIRE(tree.nodes[1].record_count() == 10);
    REQUIRE(tree.nodes[2].record_count() == 100000);

    tree.refine(0);

    tree.record_bulk(point3(-0.5F, 0.F, 0.F), spectral::ZERO(), norm_vec3(), 10,
                     SampledLambdas::new_mock(), MaterialId(0));
    tree.record_bulk(point3(0.5F, 0.5F, 0.F), spectral::ZERO(), norm_vec3(), 20,
                     SampledLambdas::new_mock(), MaterialId(0));
    tree.record_bulk(point3(0.5F, -0.5F, 0.F), spectral::ZERO(), norm_vec3(), 30,
                     SampledLambdas::new_mock(), MaterialId(0));

    REQUIRE(tree.nodes[1].record_count() == 10);
    REQUIRE(tree.nodes[3].record_count() == 30);
    REQUIRE(tree.nodes[4].record_count() == 20);
}
