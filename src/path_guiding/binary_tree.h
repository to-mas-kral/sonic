#ifndef BINARY_TREE_H
#define BINARY_TREE_H

#include "../materials/material_id.h"
#include "../spectrum/sampled_lambdas.h"

#include <atomic>
#include <mutex>
#include <unordered_map>
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
    sample(f32 xi) const;

    f32
    pdf(f32 lambda) const;

    f32
    total_flux() const {
        return nodes[0].m_radiance;
    }

    void
    record(const SampledLambdas &lambdas, const spectral &radiance);

    std::vector<BinaryTreeNode> nodes{};
};

class MaterialTrees {
public:
    MaterialTrees() = default;

    MaterialTrees(const MaterialTrees &other) : trees(other.trees) {}

    MaterialTrees(MaterialTrees &&other) noexcept : trees(std::move(other.trees)) {}

    MaterialTrees &
    operator=(const MaterialTrees &other) {
        if (this == &other) {
            return *this;
        }

        trees = other.trees;
        return *this;
    }

    MaterialTrees &
    operator=(MaterialTrees &&other) noexcept {
        if (this == &other) {
            return *this;
        }

        trees = std::move(other.trees);
        return *this;
    }

    ~MaterialTrees() = default;

    void
    refine(const f32 SUBDIVISION_CRITERION = 0.03F) {
        std::erase_if(trees, [](auto &kv) { return kv.second.total_flux() == 0.F; });

        for (auto &[_, tree] : trees) {
            tree.refine(SUBDIVISION_CRITERION);
        }
    }

    void
    reset_flux() {
        for (auto &[_, tree] : trees) {
            tree.reset_flux();
        }
    }

    SampledLambdas
    sample(const MaterialId mat_id, const f32 xi) const {
        if (!trees.contains(mat_id)) {
            return SampledLambdas::sample_visual_importance(xi);
        }

        return trees.at(mat_id).sample(xi);
    }

    f32
    pdf(const MaterialId mat_id, const f32 lambda) const {
        if (!trees.contains(mat_id)) {
            return SampledLambdas::pdf_visual_importance(lambda);
        }

        return trees.at(mat_id).pdf(lambda);
    }

    void
    record(const MaterialId mat_id, const SampledLambdas &lambdas,
           const spectral &radiance) {
        const std::scoped_lock lock(trees_mutex);
        ensure_mat_id_present(mat_id);
        trees.at(mat_id).record(lambdas, radiance);
    }

    std::unordered_map<MaterialId, BinaryTree> trees;

private:
    void
    ensure_mat_id_present(const MaterialId mat_id) {
        if (!trees.contains(mat_id)) {
            trees.insert({mat_id, BinaryTree()});
        }
    }

    std::mutex trees_mutex;
};

#endif // BINARY_TREE_H
