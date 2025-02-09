#ifndef BINARY_TREE_H
#define BINARY_TREE_H

#include "../spectrum/sampled_lambdas.h"

#include <atomic>
#include <vector>

/// A binary tree class for sampling wavelengths.
/// Simillar to the quadtree of Practical Path Guiding.
class BinaryTreeNode {
public:
    BinaryTreeNode() = default;

    explicit BinaryTreeNode(const f32 radiance) : m_radiance(radiance) {}

    BinaryTreeNode(const BinaryTreeNode &other)
        : m_radiance(other.m_radiance.load()),
          m_children_indices(other.m_children_indices) {}

    BinaryTreeNode &
    operator=(const BinaryTreeNode &other) {
        if (this == &other) {
            return *this;
        }

        m_radiance = other.m_radiance.load();
        m_children_indices = other.m_children_indices;
        return *this;
    }

    BinaryTreeNode(BinaryTreeNode &&other) noexcept = delete;
    BinaryTreeNode &
    operator=(BinaryTreeNode &&other) noexcept = delete;

    ~BinaryTreeNode() = default;

    bool
    is_leaf() const {
        return m_children_indices[0] == 0;
    }

    void
    record_radiance(const spectral &radiance) {
        // TODO: Don't know what memory ordering...
        m_radiance.fetch_add(radiance.average());
    }

    u32
    get_child(const u8 index) const {
        return m_children_indices[index];
    }

    void
    prune_children() {
        m_children_indices = {0, 0};
    }

    std::atomic<f32> m_radiance{0.F};
    std::array<u16, 2> m_children_indices{0, 0};
};

class BinaryTree {
public:
    BinaryTree() {
        nodes.emplace_back();
        nodes[0].m_radiance = 1.F;
        refine();
        reset_flux();
    }

    void
    refine(f32 SUBDIVISION_CRITERION = 0.1F);

    void
    reset_flux() {
        for (auto &node : nodes) {
            node.m_radiance = 0.F;
        }
    }

    SampledLambdas
    sample(Sampler &sampler) const;

    f32
    pdf(f32 lambda) const;

    void
    record(const SampledLambdas &lambdas, const spectral &radiance);

    std::vector<BinaryTreeNode> nodes{};
};

#endif // BINARY_TREE_H
