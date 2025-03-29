#include "../math/samplers/sampler.h"
#include "../spectrum/spectrum_consts.h"
#include "../utils/make_array.h"
#include "binary_tree.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("binarytree record") {
    BinaryTree tree{};

    tree.nodes.resize(17);
    tree.nodes[0].m_children_indices[0] = 1;
    tree.nodes[0].m_children_indices[1] = 2;

    tree.nodes[1].m_children_indices[0] = 3;
    tree.nodes[1].m_children_indices[1] = 4;

    tree.nodes[2].m_children_indices[0] = 0;
    tree.nodes[2].m_children_indices[1] = 0;

    tree.nodes[3].m_children_indices[0] = 0;
    tree.nodes[3].m_children_indices[1] = 0;

    tree.nodes[4].m_children_indices[0] = 5;
    tree.nodes[4].m_children_indices[1] = 6;

    tree.nodes[5].m_children_indices[0] = 0;
    tree.nodes[5].m_children_indices[1] = 0;

    tree.nodes[6].m_children_indices[0] = 0;
    tree.nodes[6].m_children_indices[1] = 0;

    tree.record(
        SampledLambdas(sonic::make_array<N_SPECTRUM_SAMPLES>(800.F), spectral::ONE()),
        spectral::ONE());

    REQUIRE(tree.nodes[1].m_radiance == 0.F);
    REQUIRE(tree.nodes[2].m_radiance > 0.F);
    REQUIRE(tree.nodes[3].m_radiance == 0.F);
    REQUIRE(tree.nodes[4].m_radiance == 0.F);
    REQUIRE(tree.nodes[5].m_radiance == 0.F);
    REQUIRE(tree.nodes[6].m_radiance == 0.F);

    tree.reset_radiance();

    tree.record(
        SampledLambdas(sonic::make_array<N_SPECTRUM_SAMPLES>(800.F), spectral::ONE()),
        spectral::ONE());

    tree.record(
        SampledLambdas(sonic::make_array<N_SPECTRUM_SAMPLES>(400.F), spectral::ONE()),
        spectral::ONE());

    REQUIRE(tree.nodes[1].m_radiance > 0.F);
    REQUIRE(tree.nodes[2].m_radiance > 0.F);
    REQUIRE(tree.nodes[3].m_radiance > 0.F);
    REQUIRE(tree.nodes[4].m_radiance == 0.F);
    REQUIRE(tree.nodes[5].m_radiance == 0.F);
    REQUIRE(tree.nodes[6].m_radiance == 0.F);

    tree.reset_radiance();

    tree.record(
        SampledLambdas(sonic::make_array<N_SPECTRUM_SAMPLES>(540.F), spectral::ONE()),
        spectral::ONE());

    REQUIRE(tree.nodes[1].m_radiance > 0.F);
    REQUIRE(tree.nodes[2].m_radiance == 0.F);
    REQUIRE(tree.nodes[3].m_radiance == 0.F);
    REQUIRE(tree.nodes[4].m_radiance > 0.F);
    REQUIRE(tree.nodes[5].m_radiance == 0.F);
    REQUIRE(tree.nodes[6].m_radiance > 0.F);

    tree.reset_radiance();
}

TEST_CASE("binarytree refine") {
    BinaryTree tree{};

    tree.nodes.resize(3);
    tree.nodes[0].m_children_indices[0] = 1;
    tree.nodes[0].m_children_indices[1] = 2;
    tree.nodes[0].m_radiance = 1.F;

    tree.nodes[1].m_children_indices[0] = 0;
    tree.nodes[1].m_children_indices[1] = 0;
    tree.nodes[1].m_radiance = 0.2F;

    tree.nodes[2].m_children_indices[0] = 0;
    tree.nodes[2].m_children_indices[1] = 0;
    tree.nodes[2].m_radiance = 0.8F;

    tree.refine(0.2F);

    REQUIRE(tree.nodes[0].m_radiance == 1.F);
    REQUIRE(tree.nodes[0].m_children_indices[0] == 1);
    REQUIRE(tree.nodes[0].m_children_indices[1] == 2);

    REQUIRE(tree.nodes[1].m_radiance == 0.2F);
    REQUIRE(tree.nodes[1].is_leaf());

    REQUIRE(tree.nodes[2].m_radiance == 0.8F);
    REQUIRE(tree.nodes[2].m_children_indices[0] == 3);
    REQUIRE(tree.nodes[2].m_children_indices[1] == 4);

    REQUIRE(tree.nodes[3].m_radiance == 0.4F);
    REQUIRE(tree.nodes[3].m_children_indices[0] == 5);
    REQUIRE(tree.nodes[3].m_children_indices[1] == 6);

    REQUIRE(tree.nodes[4].m_radiance == 0.4F);
    REQUIRE(tree.nodes[4].m_children_indices[0] == 7);
    REQUIRE(tree.nodes[4].m_children_indices[1] == 8);

    REQUIRE(tree.nodes[5].m_radiance == 0.2F);
    REQUIRE(tree.nodes[5].is_leaf());

    REQUIRE(tree.nodes[6].m_radiance == 0.2F);
    REQUIRE(tree.nodes[6].is_leaf());

    REQUIRE(tree.nodes[7].m_radiance == 0.2F);
    REQUIRE(tree.nodes[7].is_leaf());

    REQUIRE(tree.nodes[8].m_radiance == 0.2F);
    REQUIRE(tree.nodes[8].is_leaf());
}

TEST_CASE("binary tree sampling pdf integrates to 1") {
    auto tree = BinaryTree();
    auto sampler = Sampler(uvec2(1, 1), uvec2(10, 10), 10);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 1024U << i; ++j) {
            tree.record(SampledLambdas::sample_visual_importance(sampler.sample()),
                        spectral::ONE());
        }

        if (i != 2) {
            tree.refine();
            tree.reset_radiance();
        }
    }

    tree.refine();

    auto dimsampler = DimensionSampler();

    constexpr i32 ITERS = 16384;
    f64 pdf_sum = 0.;
    for (int i = 0; i < ITERS; ++i) {
        const auto sample =
            tree.pdf(LAMBDA_MIN + dimsampler.sample() * (LAMBDA_RANGE - 1));
        pdf_sum += sample;
    }

    const auto integral = (LAMBDA_RANGE - 1) * pdf_sum / static_cast<f64>(ITERS);
    REQUIRE_THAT(integral, Catch::Matchers::WithinRelMatcher(1., 0.01));
}

TEST_CASE("binary tree sampling pdf matches pdf") {
    auto tree = BinaryTree();
    auto sampler = Sampler(uvec2(1, 1), uvec2(10, 10), 10);

    for (u32 i = 0; i < 3; ++i) {
        for (u32 j = 0; j < 1024U << i; ++j) {
            tree.record(SampledLambdas::sample_visual_importance(sampler.sample()),
                        spectral::ONE());
        }

        if (i != 2) {
            tree.refine();
            tree.reset_radiance();
        }
    }

    tree.refine();

    constexpr i32 ITERS = 16384;
    for (int i = 0; i < ITERS; ++i) {
        const auto lambdas = tree.sample(sampler.sample());
        const auto pdf = tree.pdf(lambdas[0]);
        REQUIRE_THAT(lambdas.pdfs[0], Catch::Matchers::WithinRelMatcher(pdf, 0.0001F));
    }
}
