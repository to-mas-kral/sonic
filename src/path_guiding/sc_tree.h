#ifndef SS_TREE_H
#define SS_TREE_H

#include "../materials/material_id.h"
#include "../math/axis.h"
#include "../math/vecmath.h"
#include "../spectrum/sampled_lambdas.h"
#include "binary_tree.h"

#include <algorithm>
#include <memory>
#include <spdlog/spdlog.h>

class Reservoir {
public:
    explicit Reservoir(const u16vec2 &coords)
        : coords(coords), sampling_binary_tree(std::make_unique<BinaryTree>()),
          recording_binary_tree(std::make_unique<BinaryTree>()) {}

    Reservoir(const Reservoir &other)
        : coords(other.coords),
          sampling_binary_tree(std::make_unique<BinaryTree>(*other.sampling_binary_tree)),
          recording_binary_tree(
              std::make_unique<BinaryTree>(*other.recording_binary_tree)) {}

    Reservoir(Reservoir &&other) noexcept
        : coords(other.coords),
          sampling_binary_tree(std::move(other.sampling_binary_tree)),
          recording_binary_tree(std::move(other.recording_binary_tree)) {}

    Reservoir &
    operator=(const Reservoir &other) {
        if (this == &other) {
            return *this;
        }
        coords = other.coords;
        sampling_binary_tree = std::make_unique<BinaryTree>(*other.sampling_binary_tree);
        recording_binary_tree =
            std::make_unique<BinaryTree>(*other.recording_binary_tree);
        return *this;
    }

    Reservoir &
    operator=(Reservoir &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        coords = other.coords;
        sampling_binary_tree = std::move(other.sampling_binary_tree);
        recording_binary_tree = std::move(other.recording_binary_tree);
        return *this;
    }

    ~Reservoir() = default;

    SampledLambdas
    sample(const f32 xi) const {
        return sampling_binary_tree->sample(xi);
    }

    f32
    pdf(const f32 lambda) const {
        return sampling_binary_tree->pdf(lambda);
    }

    void
    record(const SampledLambdas &lambdas, const spectral &radiance) const {
        recording_binary_tree->record(lambdas, radiance);
    }

    void
    refine() {
        recording_binary_tree->refine(0.03F);
        sampling_binary_tree = std::make_unique<BinaryTree>(*recording_binary_tree);
        recording_binary_tree->reset_flux();
    }

    u16vec2 coords;

    std::unique_ptr<BinaryTree> sampling_binary_tree;
    std::unique_ptr<BinaryTree> recording_binary_tree;
};

constexpr u8 NUM_RESERVOIRS_SPLIT = 8;
constexpr u8 MAX_RESERVOIRS_IN_NODE = 64;

class ScTreeNode {
public:
    ScTreeNode() { reservoirs.reserve(MAX_RESERVOIRS_IN_NODE); }

    ScTreeNode(std::vector<Reservoir> &&reservoirs, const Axis split_axis,
               const u16 split_coord)
        : reservoirs(std::move(reservoirs)), split_axis(split_axis),
          split_coord(split_coord) {
        this->reservoirs.reserve(MAX_RESERVOIRS_IN_NODE);
    }

    ScTreeNode(const ScTreeNode &other)
        : reservoirs(other.reservoirs), split_axis(other.split_axis),
          split_coord(other.split_coord), first_child_index(other.first_child_index),
          second_child_index(other.second_child_index) {}

    ScTreeNode(ScTreeNode &&other) noexcept
        : reservoirs(std::move(other.reservoirs)), split_axis(other.split_axis),
          split_coord(other.split_coord), first_child_index(other.first_child_index),
          second_child_index(other.second_child_index) {}

    ScTreeNode &
    operator=(const ScTreeNode &other) {
        if (this == &other) {
            return *this;
        }
        reservoirs = other.reservoirs;
        split_axis = other.split_axis;
        split_coord = other.split_coord;
        first_child_index = other.first_child_index;
        second_child_index = other.second_child_index;
        return *this;
    }

    ScTreeNode &
    operator=(ScTreeNode &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        reservoirs = std::move(other.reservoirs);
        split_axis = other.split_axis;
        split_coord = other.split_coord;
        first_child_index = other.first_child_index;
        second_child_index = other.second_child_index;
        return *this;
    }

    ~ScTreeNode() = default;

    Reservoir &
    find_reservoir(const uvec2 pixel) {
        // TODO: have a bool flag for when inserting should not be done anymore so mutex
        // doesn't need to be locked
        const std::scoped_lock lock(reservoirs_mutex);

        auto min_dist = std::numeric_limits<f32>::max();
        auto min_index = 0;

        for (u32 i = 0; i < reservoirs.size(); ++i) {
            const auto &reservoir = reservoirs[i];

            const auto res_coord = vec2(reservoir.coords.x, reservoir.coords.y);
            const auto pixel_coord = vec2(pixel.x, pixel.y);
            const auto dist = (res_coord - pixel_coord).length();
            if (dist < min_dist) {
                min_dist = dist;
                min_index = i;
            }
        }

        assert(reservoirs.capacity() == MAX_RESERVOIRS_IN_NODE);
        if (min_dist < 50.F || reservoirs.size() >= MAX_RESERVOIRS_IN_NODE) {
            return reservoirs[min_index];
        } else {
            // Need to insert reservoir because the closest one is too far
            auto reservoir = Reservoir(u16vec2(pixel.x, pixel.y));
            reservoirs.push_back(reservoir);
            return reservoirs.back();
        }
    }

    void
    refine_reservoirs() {
        for (auto &reservoir : reservoirs) {
            reservoir.refine();
        }
    }

    bool
    is_leaf() const {
        return first_child_index == 0 && second_child_index == 0;
    }

    std::vector<Reservoir> reservoirs;
    std::mutex reservoirs_mutex;

    Axis split_axis;
    u16 split_coord;

    // First child = smaller or equal than split axis
    u16 first_child_index{0};
    // Second child = bigger than split axis
    u16 second_child_index{0};
};

class ScTree {
public:
    ScTree() { nodes.push_back(ScTreeNode()); }

    Reservoir &
    find_reservoir(const uvec2 pixel) {
        u32 node_index = 0;

        while (true) {
            auto &node = nodes[node_index];
            if (node.is_leaf()) {
                return node.find_reservoir(pixel);
            }

            if (node.split_axis == Axis::X) {
                if (pixel.x > node.split_coord) {
                    node_index = node.second_child_index;
                } else {
                    node_index = node.first_child_index;
                }
            } else {
                if (pixel.y > node.split_coord) {
                    node_index = node.second_child_index;
                } else {
                    node_index = node.first_child_index;
                }
            }
        }
    }

    void
    refine() {
        u32 i = 0;
        while (i < nodes.size()) {
            auto &node = nodes[i];
            if (node.is_leaf()) {
                node.refine_reservoirs();

                if (node.reservoirs.size() > NUM_RESERVOIRS_SPLIT) {
                    std::vector<Reservoir> first_half_reservoirs;
                    first_half_reservoirs.reserve(nodes.size());
                    std::vector<Reservoir> second_half_reservoirs;
                    first_half_reservoirs.reserve(nodes.size());

                    // Need to split reservoirs into child nodes
                    // Choose axes perpendicular to the longest extent

                    // Compute bounding box
                    u16vec2 lower{std::numeric_limits<u16>::max()};
                    u16vec2 upper{std::numeric_limits<u16>::min()};

                    for (auto &reservoir : node.reservoirs) {
                        auto coords = reservoir.coords;
                        upper.x = std::max(coords.x, upper.x);
                        upper.y = std::max(coords.y, upper.y);
                        lower.x = std::min(coords.x, lower.x);
                        lower.y = std::min(coords.y, lower.y);
                    }

                    Axis split_axis;
                    u16 split_coord;
                    if (upper.x - lower.x > upper.y - lower.y) {
                        split_axis = Axis::X;
                        split_coord = lower.x + (upper.x - lower.x) / 2;
                    } else {
                        split_axis = Axis::Y;
                        split_coord = lower.y + (upper.y - lower.y) / 2;
                    }

                    for (auto &reservoir : node.reservoirs) {
                        auto coords = reservoir.coords;
                        if (split_axis == Axis::X) {
                            if (coords.x > split_coord) {
                                second_half_reservoirs.push_back(std::move(reservoir));
                            } else {
                                first_half_reservoirs.push_back(std::move(reservoir));
                            }
                        } else {
                            if (coords.y > split_coord) {
                                second_half_reservoirs.push_back(std::move(reservoir));
                            } else {
                                first_half_reservoirs.push_back(std::move(reservoir));
                            }
                        }
                    }

                    node.reservoirs.clear();
                    auto first_node = ScTreeNode(std::move(first_half_reservoirs),
                                                 split_axis, split_coord);
                    auto second_node = ScTreeNode(std::move(second_half_reservoirs),
                                                  split_axis, split_coord);

                    node.first_child_index = nodes.size();
                    node.second_child_index = nodes.size() + 1;

                    nodes.push_back(std::move(first_node));
                    nodes.push_back(std::move(second_node));
                }
            }

            i++;
        }
    }

private:
    std::vector<ScTreeNode> nodes;
};

constexpr u32 MAX_RESERVOIRS_IN_VEC = 256;

class Reservoirs {
public:
    Reservoirs() { reservoirs.reserve(MAX_RESERVOIRS_IN_VEC); }

    Reservoirs(const Reservoirs &other) : reservoirs(other.reservoirs) {}

    Reservoirs(Reservoirs &&other) noexcept : reservoirs(std::move(other.reservoirs)) {}

    Reservoirs &
    operator=(const Reservoirs &other) {
        if (this == &other) {
            return *this;
        }
        reservoirs = other.reservoirs;
        return *this;
    }
    Reservoirs &
    operator=(Reservoirs &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        reservoirs = std::move(other.reservoirs);
        return *this;
    }

    ~Reservoirs() = default;

    Reservoir &
    find_reservoir(const uvec2 pixel);

    void
    refine() {
        for (auto &res : reservoirs) {
            res.refine();
        }
    }

    std::vector<Reservoir> reservoirs;
    std::mutex reservoirs_mutex{};
};

class LgTree {
public:
    LgTree() = default;

    LgTree(const LgTree &other) : m_reservoirs(other.m_reservoirs) {}

    LgTree(LgTree &&other) noexcept : m_reservoirs(std::move(other.m_reservoirs)) {}

    LgTree &
    operator=(const LgTree &other) {
        if (this == &other) {
            return *this;
        }
        m_reservoirs = other.m_reservoirs;
        return *this;
    }

    LgTree &
    operator=(LgTree &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        m_reservoirs = std::move(other.m_reservoirs);
        return *this;
    }

    ~LgTree() = default;

    Reservoir &
    find_reservoir(const MaterialId mat_id, const uvec2 pixel) {
        {
            const std::scoped_lock lock(m_trees_mutex);
            if (!m_reservoirs.contains(mat_id)) {
                m_reservoirs.insert({mat_id, Reservoirs()});
            }
        }

        return m_reservoirs.at(mat_id).find_reservoir(pixel);
    }

    void
    refine() {
        for (auto &kv : m_reservoirs) {
            kv.second.refine();
        }
    }

    const std::unordered_map<MaterialId, Reservoirs> &
    reservoirs() const {
        return m_reservoirs;
    }

private:
    std::unordered_map<MaterialId, Reservoirs> m_reservoirs;
    std::mutex m_trees_mutex;
};

#endif // SS_TREE_H
