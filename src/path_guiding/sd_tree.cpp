#include "sd_tree.h"

#include "../scene/sphere_square_mapping.h"
#include "../utils/panic.h"
#include "../utils/sampler.h"

u32
SDTreeNode::traverse(const point3 &pos, const Axis split_axis, AABB &bounds) const {
    assert(bounds.contains(pos));
    if (bounds.right_of_split_axis(pos, split_axis)) {
        bounds = bounds.right_half(split_axis);
        return right_child();
    } else {
        bounds = bounds.left_half(split_axis);
        return left_child();
    }
}

u32
QuadtreeNode::choose_child(const vec2 &xy, vec2 &middle, f32 &quadtrant_half) const {
    assert(middle.x >= 0.f && middle.x <= 1.f);
    assert(middle.y >= 0.f && middle.y <= 1.f);

    const auto old_quadrant_half = quadtrant_half;
    quadtrant_half /= 2.f;

    if (xy.x > middle.x) {
        middle.x += old_quadrant_half;
        if (xy.y > middle.y) {
            // bottom-right
            middle.y += old_quadrant_half;
            return m_children[1];
        } else {
            // top-right
            middle.y -= old_quadrant_half;
            return m_children[0];
        }
    } else {
        middle.x -= old_quadrant_half;
        if (xy.y > middle.y) {
            // bottom-left
            middle.y += old_quadrant_half;
            return m_children[2];
        } else {
            // top-left
            middle.y -= old_quadrant_half;
            return m_children[3];
        }
    }
}

void
Quadtree::record(const spectral &radiance, const norm_vec3 &wi) {
    const auto xy = sphere_to_square(wi);

    auto middle = vec2(0.5f);
    f32 quadtrant_half = 0.5f / 2.f;

    u32 index = 0;
    while (true) {
        auto &node = nodes[index];

        node.record_radiance(radiance);

        if (node.is_leaf()) {
            break;
        }

        index = node.choose_child(xy, middle, quadtrant_half);
    }
}

PGSample
Quadtree::sample(Sampler &sampler) const {
    auto middle = vec2(0.5f);
    f32 quadrant_len_half = 0.5f / 2.f;
    f32 inverse_square_area = 4.f;
    f32 square_probability = 1.f;

    u32 index = 0;
    while (true) {
        const auto &node = nodes[index];

        // If it's root node with 0 energy, sample directly...
        if (node.is_leaf() || (node.m_radiance == 0.f && index == 0)) {
            const auto offset = sampler.sample2() * quadrant_len_half * 4.f;
            const auto top_left = middle - vec2(quadrant_len_half * 2.f);
            const auto xy = top_left + offset;

            return PGSample{
                .wi = square_to_sphere(xy),
                .pdf = square_probability * (1.f / (4.f * M_PIf)),
            };
        }

        const auto &c0_rad = nodes[node.child_index(0)].m_radiance.load();
        const auto &c1_rad = nodes[node.child_index(1)].m_radiance.load();
        const auto &c2_rad = nodes[node.child_index(2)].m_radiance.load();
        const auto &c3_rad = nodes[node.child_index(3)].m_radiance.load();

        const auto sum = c0_rad + c1_rad + c2_rad + c3_rad;

        assert(sum > 0.f);

        const std::array<f32, 4> cdf = {
            c0_rad / sum,
            (c0_rad + c1_rad) / sum,
            (c0_rad + c1_rad + c2_rad) / sum,
            1.f,
        };

        const auto xi = sampler.sample();

        auto sampled_child = 0;
        if (xi < cdf[0]) {
            sampled_child = 0;
            square_probability *= 4.f * c0_rad / sum;
            middle.x += quadrant_len_half;
            middle.y -= quadrant_len_half;
        } else if (xi < cdf[1]) {
            sampled_child = 1;
            square_probability *= 4.f * c1_rad / sum;
            middle.x += quadrant_len_half;
            middle.y += quadrant_len_half;
        } else if (xi < cdf[2]) {
            sampled_child = 2;
            square_probability *= 4.f * c2_rad / sum;
            middle.x -= quadrant_len_half;
            middle.y += quadrant_len_half;
        } else {
            square_probability *= 4.f * c3_rad / sum;
            sampled_child = 3;
            middle.x -= quadrant_len_half;
            middle.y -= quadrant_len_half;
        }

        quadrant_len_half /= 2.f;
        inverse_square_area *= 4.f;

        index = node.child_index(sampled_child);
    }
}

void
Quadtree::refine() {
    const f32 total_flux = nodes[0].m_radiance.load();
    constexpr f32 SUBDIVIDE_CRITERION = 0.01f;

    std::vector<QuadtreeNode> new_nodes{};
    new_nodes.resize(nodes.size());

    struct TraversalIndex {
        u32 index;
        u32 insert_at;
    };

    {
        std::vector stack = {TraversalIndex{
            .index = 0,
            .insert_at = 0,
        }};

        // Prune the children first
        u32 size_counter = 1;
        while (!stack.empty()) {
            const auto trav = stack.back();
            auto node = nodes[trav.index];
            stack.pop_back();

            if (!node.is_leaf()) {
                if (node.m_radiance.load() / total_flux < SUBDIVIDE_CRITERION) {
                    // Prune children = don't push them to the stack
                    node.prune_children();
                } else {
                    for (int i = 0; i < 4; ++i) {
                        const auto insert_at = size_counter;
                        assert(insert_at < new_nodes.size());
                        stack.push_back(TraversalIndex{.index = node.m_children[i],
                                                       .insert_at = insert_at});
                        node.m_children[i] = insert_at;
                        size_counter++;
                    }
                }
            }

            new_nodes[trav.insert_at] = node;
        }

        new_nodes.resize(size_counter);
    }

    // Subdivide leafs recursively
    u32 i = 0;
    while (i < new_nodes.size()) {
        auto &node = new_nodes[i];
        const auto node_flux = node.m_radiance.load();
        if (node.is_leaf() && node_flux / total_flux > SUBDIVIDE_CRITERION) {
            node.m_children[0] = new_nodes.size();
            node.m_children[1] = new_nodes.size() + 1;
            node.m_children[2] = new_nodes.size() + 2;
            node.m_children[3] = new_nodes.size() + 3;

            new_nodes.emplace_back(node_flux / 4.f);
            new_nodes.emplace_back(node_flux / 4.f);
            new_nodes.emplace_back(node_flux / 4.f);
            new_nodes.emplace_back(node_flux / 4.f);
        }

        i++;
    }

    // new_nodes.shrink_to_fit()...

    nodes = new_nodes;
}

void
SDTree::record(const point3 &pos, const spectral &radiance, const norm_vec3 &wi) {
    auto &node = traverse<nullptr>(pos);
    node.record(radiance, wi);
}

PGSample
SDTree::sample(const point3 &pos, Sampler &sampler) {
    const auto &node = traverse<nullptr>(pos);

    auto sampler_copy = sampler;

    const auto s = node.sample(sampler);
    if (std::isnan(s.pdf) || std::isinf(s.pdf)) {
        auto a = node.sample(sampler_copy);
    }
    assert(!std::isnan(s.pdf) && !std::isinf(s.pdf));

    return s;
}

void
SDTree::refine(const u32 iteration) {
    constexpr f32 C = 12000.f; // constant from the paper
    const f32 SUBDIVIDE_CRITERION = C * sqrtf(1 << (iteration + 1));

    // nodes vector is being modified, iterate by index
    const auto initial_size = nodes.size();
    for (u32 i = 0; i < initial_size; ++i) {
        auto &node = nodes[i];
        if (node.is_leaf()) {
            if (node.record_count() > SUBDIVIDE_CRITERION) {
                if (node.parent_index() != 0) {
                    const auto &parent = nodes[node.parent_index()];
                    node.set_split_axis(next_axis(parent.split_axis()));
                }
                node.set_children(nodes.size(), nodes.size() + 1);

                auto l_child_node = SDTreeNode(i);
                l_child_node.m_recording_quadtree =
                    std::make_unique<Quadtree>(*node.m_recording_quadtree);

                /*l_child_node.m_sampling_quadtree =
                    std::make_unique<Quadtree>(*node.m_sampling_quadtree);*/

                auto r_child_node = SDTreeNode(i);
                r_child_node.m_recording_quadtree = std::move(node.m_recording_quadtree);
                // Move the sampling tree as well to set it to nullptr in the parent node
                r_child_node.m_sampling_quadtree = std::move(node.m_sampling_quadtree);

                nodes.push_back(l_child_node);
                nodes.push_back(r_child_node);
            }
        } else {
            assert(node.record_count() == 0);
        }
    }

    // New leaf nodes have been added, refine the quadtrees here
    for (auto &node : nodes) {
        node.reset_count();

        if (node.is_leaf()) {
            node.m_recording_quadtree->refine();
            node.m_sampling_quadtree =
                std::make_unique<Quadtree>(*node.m_recording_quadtree);

            node.m_recording_quadtree->reset_flux();
        } else {
            assert(node.m_recording_quadtree == nullptr);
            assert(node.m_sampling_quadtree == nullptr);
        }
    }
}

template <void (*NODE_VISITOR)(SDTreeNode &), typename... Ts>
SDTreeNode &
SDTree::traverse(const point3 &pos, Ts... args) {
    u32 index = 0;
    auto split_axis = Axis::X;
    auto bounds = scene_bounds;

    while (true) {
        auto &node = nodes[index];
        assert(node.contains(pos, bounds));

        if constexpr (NODE_VISITOR != nullptr) {
            NODE_VISITOR(node, args...);
        }

        if (node.is_leaf()) {
            return node;
        }

        index = node.traverse(pos, split_axis, bounds);

        split_axis = next_axis(split_axis);
    }
}
