#include "binary_tree.h"

#include "../spectrum/spectrum.h"
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
BinaryTree::sample(const f32 xi_base) const {
    std::array<f32, N_SPECTRUM_SAMPLES> lambdas;
    spectral pdfs;

    if (nodes[0].m_radiance == 0.F) {
        return SampledLambdas::sample_visual_importance(xi_base);
    }

    for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
        auto xi = xi_base + static_cast<f32>(i) / static_cast<f32>(N_SPECTRUM_SAMPLES);
        if (xi > 1.F) {
            xi -= 1.F;
        }

        u32 node_index = 0;
        f32 midpoint = 0.5F;
        f32 current_half = 0.25F;
        f32 node_probability = 1.F;
        while (true) {
            const auto &node = nodes[node_index];

            if (node.is_leaf() || node.m_radiance == 0.F) {
                const auto node_start = midpoint - 2.F * current_half;
                const auto node_end = midpoint + 2.F * current_half;

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

            if (xi < r_child_frac) {
                midpoint += current_half;
                node_index = node.m_children_indices[1];
                node_probability *= 2.F * r_child_frac;
                xi /= r_child_frac;
            } else {
                midpoint -= current_half;
                node_index = node.m_children_indices[0];
                node_probability *= 2.F * l_child_frac;
                xi -= r_child_frac;
                xi /= l_child_frac;
            }

            current_half /= 2.F;
        }
    }

    return SampledLambdas(lambdas, pdfs);
}

f32
BinaryTree::pdf(const f32 lambda) const {
    const f32 lambda_map = (lambda - LAMBDA_MIN) / (LAMBDA_RANGE - 1);

    if (nodes[0].m_radiance == 0.F) {
        return SampledLambdas::pdf_visual_importance(lambda);
    }

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
    spectral contributions = radiance;
    for (u32 i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
        const auto sensor_response = SampledLambdas::pdf_visual_importance(lambdas[i]);

        if (contributions[i] != 0.F) {
            contributions[i] *= sensor_response / lambdas.pdfs[i];

            if (lambdas.weights[i] != -1.F) {
                contributions[i] *= lambdas.weights[i];
            }
        }
    }

    spectral lambdas_mapped;
    for (u32 i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
        lambdas_mapped[i] = (lambdas[i] - LAMBDA_MIN) / (LAMBDA_RANGE - 1);
    }

    struct TraversalNode {
        f32 midpoint;
        f32 current_half;
        u32 node_index;
        u8 lambdas_mask;
    };

    constexpr u64 TRAV_NODE_BUFFER_SIZE = 64;
    std::array<TraversalNode, TRAV_NODE_BUFFER_SIZE> buffer;
    buffer[0] = TraversalNode{
        .midpoint = 0.5F,
        .current_half = 0.25F,
        .node_index = 0,
        .lambdas_mask = 0b1111'1111,
    };
    u32 buf_size = 1;

    constexpr std::array<u8, 8> masks = {1U,       1U << 1U, 1U << 2U, 1U << 3U,
                                         1U << 4U, 1U << 5U, 1U << 6U, 1U << 7U};

    for (u32 i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
        if (contributions[i] == 0) {
            buffer[0].lambdas_mask ^= masks[i];
        }
    }

    while (buf_size > 0) {
        const auto trav_node = buffer[buf_size - 1];
        buf_size--;

        auto &node = nodes[trav_node.node_index];

        {
            f32 contribution = 0.F;
            for (u32 i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
                if (trav_node.node_index == 0 ||
                    (trav_node.lambdas_mask & masks[i]) != 0) {
                    contribution += contributions[i];
                }
            }

            node.m_radiance.fetch_add(contribution, std::memory_order_relaxed);
        }

        if (node.is_leaf()) {
            continue;
        }

        // Setup traversal for left node
        {
            u8 lambdas_mask = trav_node.lambdas_mask;
            for (u32 i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
                if ((lambdas_mask & masks[i]) != 0) {
                    // If lambda belongs to the right subtree, delete it from the mask
                    if (lambdas_mapped[i] > trav_node.midpoint) {
                        lambdas_mask ^= masks[i];
                    }
                }
            }

            if (lambdas_mask != 0) {
                buffer[buf_size] = TraversalNode{
                    .midpoint = trav_node.midpoint - trav_node.current_half,
                    .current_half = trav_node.current_half / 2.F,
                    .node_index = node.m_children_indices[0],
                    .lambdas_mask = lambdas_mask,
                };

                buf_size++;
            }
        }

        // Setup traversal for right node
        {
            u8 lambdas_mask = trav_node.lambdas_mask;
            for (u32 i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
                if ((lambdas_mask & masks[i]) != 0) {
                    // If lambda belongs to the right subtree, delete it from the mask
                    if (lambdas_mapped[i] <= trav_node.midpoint) {
                        lambdas_mask ^= masks[i];
                    }
                }
            }

            if (lambdas_mask != 0) {
                buffer[buf_size] = TraversalNode{
                    .midpoint = trav_node.midpoint + trav_node.current_half,
                    .current_half = trav_node.current_half / 2.F,
                    .node_index = node.m_children_indices[1],
                    .lambdas_mask = lambdas_mask,
                };

                buf_size++;
            }
        }

        if (buf_size > TRAV_NODE_BUFFER_SIZE) {
            panic("Binary tree traversal buffer overflow");
        }
    }

    /*for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
        if (radiance[i] == 0.F) {
            continue;
        }

        const f32 lambda = lambdas[i];
        const f32 lambda_map = (lambda - LAMBDA_MIN) / (LAMBDA_RANGE - 1);

        u32 node_index = 0;
        f32 midpoint = 0.5F;
        f32 current_half = 0.25F;
        while (true) {
            auto &node = nodes[node_index];

            const auto sensor_response =
                SampledLambdas::pdf_visual_importance(lambdas[i]);

            node.m_radiance.fetch_add(radiance[i] * sensor_response / lambdas.pdfs[i],
                                      std::memory_order_relaxed);

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
    }*/
}