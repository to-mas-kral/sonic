#ifndef SD_TREE_H
#define SD_TREE_H

#include "../color/spectral_quantity.h"
#include "../math/aabb.h"
#include "../math/vecmath.h"
#include "../utils/basic_types.h"

#include <vector>

class Sampler;

struct PGSample {
    norm_vec3 wi;
    f32 pdf;
};

class QuadtreeNode {
public:
    QuadtreeNode() = default;

    explicit QuadtreeNode(const f32 radiance) : m_radiance(radiance) {}

    QuadtreeNode(const QuadtreeNode &other)
        : m_radiance(other.m_radiance.load()), m_children(other.m_children) {}

    QuadtreeNode &
    operator=(const QuadtreeNode &other) {
        if (this == &other) {
            return *this;
        }

        m_radiance = other.m_radiance.load();
        m_children = other.m_children;
        return *this;
    }

    QuadtreeNode(QuadtreeNode &&other) noexcept = delete;
    QuadtreeNode &
    operator=(QuadtreeNode &&other) noexcept = delete;

    ~QuadtreeNode() = default;

    bool
    is_leaf() const {
        return m_children[0] == 0;
    }

    void
    record_radiance(const spectral &radiance) {
        // TODO: Don't know what memory ordering...
        m_radiance.fetch_add(radiance.average());
    }

    /// Returns the index inside of the m_children.
    u32
    choose_child(const vec2 &xy, vec2 &middle, f32 &quadrant_half) const;

    u32
    child_index(const u8 index) const {
        return m_children[index];
    }

    void
    prune_children() {
        m_children = {0, 0, 0, 0};
    }

    std::atomic<f32> m_radiance{0.F};

    /// Odrer of children:
    ///
    /// 0,0 |   x   0.5     1
    ///   --.--------------->
    ///  y  |   3   |   0   |
    ///     |       |       |
    /// 0.5 |-------|-------|
    ///     |   2   |   1   |
    ///     |       |       |
    ///  1  |-------|-------|
    std::array<u16, 4> m_children{0, 0, 0, 0};
};

class Quadtree {
public:
    Quadtree() { nodes.emplace_back(); }

    void
    record(const spectral &radiance, const norm_vec3 &wi);

    PGSample
    sample(Sampler &sampler) const;

    void
    refine();

    void
    reset_flux() {
        for (auto &node : nodes) {
            node.m_radiance = 0.F;
        }
    }

    /*#ifdef TEST_PUBLIC
    public:
    #else
    private:
    #endif*/
    std::vector<QuadtreeNode> nodes{};
};

class SDTreeNode {
public:
    explicit SDTreeNode(const u32 parent_index)
        : m_recording_quadtree{std::make_unique<Quadtree>()},
          m_sampling_quadtree{std::make_unique<Quadtree>()},
          m_parent_index{parent_index} {}

    SDTreeNode(const SDTreeNode &other)
        : m_split_axis(other.m_split_axis), m_parent_index(other.m_parent_index),
          m_left_child(other.m_left_child), m_right_child(other.m_right_child),
          m_record_count(other.m_record_count.load()) {
        if (other.m_recording_quadtree) {
            m_recording_quadtree =
                std::make_unique<Quadtree>(*other.m_recording_quadtree);
        }
        if (other.m_sampling_quadtree) {
            m_sampling_quadtree = std::make_unique<Quadtree>(*other.m_sampling_quadtree);
        }
    }

    SDTreeNode &
    operator=(const SDTreeNode &other) = delete;

    SDTreeNode(SDTreeNode &&other) noexcept = delete;
    SDTreeNode &
    operator=(SDTreeNode &&other) noexcept = delete;

    // I don't think any custom destructor is needed ?
    ~SDTreeNode() = default;

    bool
    is_leaf() const {
        return m_right_child == 0;
    }

    u32
    left_child() const {
        return m_left_child;
    }

    u32
    right_child() const {
        return m_right_child;
    }

    void
    set_children(const u32 l_child_index, const u32 r_child_index) {
        m_left_child = l_child_index;
        m_right_child = r_child_index;
    }

    u32
    parent_index() const {
        return m_parent_index;
    }

    /// Returns the next child index
    /// sets bounds to the bounds of the respective child node
    u32
    traverse(const point3 &pos, Axis split_axis, AABB &bounds) const;

    void
    split(Axis axis);

    void
    record(const spectral &radiance, const norm_vec3 &wi) {
        m_record_count.fetch_add(1, std::memory_order_relaxed);
        m_recording_quadtree->record(radiance, wi);
    }

    void
    record_bulk(const spectral &radiance, const norm_vec3 &wi, const u32 count) {
        m_record_count.fetch_add(count, std::memory_order_relaxed);
        m_recording_quadtree->record(radiance, wi);
    }

    PGSample
    sample(Sampler &sampler) const {
        return m_sampling_quadtree->sample(sampler);
    }

    u32
    record_count() {
        return m_record_count;
    }

    void
    reset_count() {
        m_record_count = 0;
    }

    Axis
    split_axis() const {
        return m_split_axis;
    }

    void
    set_split_axis(const Axis axis) {
        m_split_axis = axis;
    }

    static bool
    contains(const point3 &pos, const AABB &bounds) {
        return bounds.contains(pos);
    }

    std::unique_ptr<Quadtree> m_recording_quadtree{nullptr};
    std::unique_ptr<Quadtree> m_sampling_quadtree{nullptr};

private:
    //  TODO: could put this inside a union
    Axis m_split_axis{Axis::X};
    u32 m_parent_index{0};
    u32 m_left_child{0};
    u32 m_right_child{0};
    std::atomic_uint32_t m_record_count{0};
};

class SDTree {
public:
    explicit SDTree(const AABB &scene_bounds) : scene_bounds{scene_bounds} {
        nodes.emplace_back(0);
    }

    void
    record(const point3 &pos, const spectral &radiance, const norm_vec3 &wi);

    void
    record_bulk(const point3 &pos, const spectral &radiance, const norm_vec3 &wi,
                u32 count);

    PGSample
    sample(const point3 &pos, Sampler &sampler);

    /// Creates a new sampling tree from the recording tree
    /// iteration starts from 0 as far as I can tell
    void
    refine(u32 iteration);

private:
    template <void (*NODE_VISITOR)(SDTreeNode &), typename... Ts>
    SDTreeNode &
    traverse(const point3 &pos, Ts...);

    AABB scene_bounds;
    std::pmr::vector<SDTreeNode> nodes{};
};

#endif // SD_TREE_H
