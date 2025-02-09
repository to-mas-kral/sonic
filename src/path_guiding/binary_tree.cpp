#include "binary_tree.h"

#include "../spectrum/spectrum_consts.h"

void
BinaryTree::refine(const f32 SUBDIVISION_CRITERION) {
    const f32 total_radiance = nodes[0].m_radiance.load();

    std::vector<BinaryTreeNode> new_nodes{};
    new_nodes.resize(nodes.size());

    struct TraversalIndex {
        u32 index_in_old;
        u32 index_in_new;
    };

    std::vector stack = {TraversalIndex{
        .index_in_old = 0,
        .index_in_new = 0,
    }};

    // Prune the children first
    u32 size_counter = 1;
    while (!stack.empty()) {
        const auto trav = stack.back();
        auto node = nodes[trav.index_in_old];
        stack.pop_back();

        if (!node.is_leaf()) {
            if (node.m_radiance.load() / total_radiance < SUBDIVISION_CRITERION) {
                // Prune children = don't push them to the stack
                node.prune_children();
            } else {
                for (int i = 0; i < 2; ++i) {
                    const auto index_in_new = size_counter;
                    assert(index_in_new < new_nodes.size());
                    stack.push_back(
                        TraversalIndex{.index_in_old = node.m_children_indices[i],
                                       .index_in_new = index_in_new});
                    node.m_children_indices[i] = index_in_new;
                    size_counter++;
                }
            }
        }

        new_nodes[trav.index_in_new] = node;
    }

    new_nodes.resize(size_counter);

    // Subdivide leafs recursively
    u32 i = 0;
    while (i < new_nodes.size()) {
        auto &node = new_nodes[i];
        const auto node_flux = node.m_radiance.load();
        if (node.is_leaf() && node_flux / total_radiance > SUBDIVISION_CRITERION) {
            node.m_children_indices[0] = new_nodes.size();
            node.m_children_indices[1] = new_nodes.size() + 1;

            new_nodes.emplace_back(node_flux / 2.F);
            new_nodes.emplace_back(node_flux / 2.F);
        }

        i++;
    }

    // new_nodes.shrink_to_fit()...
    // TODO: move
    nodes = new_nodes;
}

SampledLambdas
BinaryTree::sample(Sampler &sampler) const {
    std::array<f32, N_SPECTRUM_SAMPLES> lambdas;
    spectral pdfs;

    if (nodes[0].m_radiance == 0.F) {
        return SampledLambdas::new_sample_importance(sampler);
    }

    for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
        u32 node_index = 0;
        f32 midpoint = 0.5F;
        f32 current_half = 0.25F;
        f32 node_probability = 1.F;
        while (true) {
            auto &node = nodes[node_index];

            if (node.is_leaf() || node.m_radiance == 0.F) {
                const auto node_start = midpoint - 2.F * current_half;
                const auto node_end = midpoint + 2.F * current_half;

                const auto xi = sampler.sample();
                const auto lambda_map = node_start + xi * (node_end - node_start);

                const auto lambda = LAMBDA_MIN + lambda_map * (LAMBDA_RANGE - 1);

                lambdas[i] = lambda;
                pdfs[i] = node_probability / static_cast<f32>(LAMBDA_RANGE - 1);

                break;
            }

            const auto &r_child = nodes[node.m_children_indices[1]];
            const auto &l_child = nodes[node.m_children_indices[0]];

            const auto radiance_sum = r_child.m_radiance + l_child.m_radiance;
            const auto r_child_frac = r_child.m_radiance / radiance_sum;
            const auto l_child_frac = l_child.m_radiance / radiance_sum;

            const f32 xi = sampler.sample();
            if (xi < r_child_frac) {
                midpoint += current_half;
                node_index = node.m_children_indices[1];
                node_probability *= 2.F * r_child_frac;
            } else {
                midpoint -= current_half;
                node_index = node.m_children_indices[0];
                node_probability *= 2.F * l_child_frac;
            }

            current_half /= 2.F;
        }
    }

    return SampledLambdas(lambdas, pdfs);
}

f32
BinaryTree::pdf(const f32 lambda) const {
    const f32 lambda_map = (lambda - LAMBDA_MIN) / (LAMBDA_RANGE - 1);

    u32 node_index = 0;
    f32 midpoint = 0.5F;
    f32 current_half = 0.25F;
    f32 node_probability = 1.F;
    while (true) {
        const auto &node = nodes[node_index];

        if (node.is_leaf() || node.m_radiance == 0.F) {
            return node_probability / static_cast<f32>(LAMBDA_RANGE - 1);
        }

        const auto &r_child = nodes[node.m_children_indices[1]];
        const auto &l_child = nodes[node.m_children_indices[0]];

        const auto radiance_sum = r_child.m_radiance + l_child.m_radiance;
        const auto r_child_frac = r_child.m_radiance / radiance_sum;
        const auto l_child_frac = l_child.m_radiance / radiance_sum;

        if (lambda_map > midpoint) {
            midpoint += current_half;
            node_index = node.m_children_indices[1];
            node_probability *= 2.F * r_child_frac;
        } else {
            midpoint -= current_half;
            node_index = node.m_children_indices[0];
            node_probability *= 2.F * l_child_frac;
        }

        current_half /= 2.F;
    }
}

void
BinaryTree::record(const SampledLambdas &lambdas, const spectral &radiance) {
    // TODO: this is O(n^2), improve later
    for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
        const f32 lambda = lambdas[i];
        const f32 lambda_map = (lambda - LAMBDA_MIN) / (LAMBDA_RANGE - 1);

        u32 node_index = 0;
        f32 midpoint = 0.5F;
        f32 current_half = 0.25F;
        while (true) {
            auto &node = nodes[node_index];

            node.m_radiance.fetch_add(radiance[i] / lambdas.pdfs[i]);

            if (node.is_leaf()) {
                break;
            }

            if (lambda_map > midpoint) {
                midpoint += current_half;
                node_index = node.m_children_indices[1];
            } else {
                midpoint -= current_half;
                node_index = node.m_children_indices[0];
            }

            current_half /= 2.F;
        }
    }
}