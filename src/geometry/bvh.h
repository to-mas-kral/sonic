#ifndef PT_BVH_H
#define PT_BVH_H

#include <span>

#include "aabb.h"
#include "axis.h"
#include "geometry.h"
#include "intersection.h"

/*
 * BVH implementation was taken from PBRTv4 !
 * */

// LinearBVHNode Definition
struct LinearBVHNode {
    AABB aabb;
    union {
        int primitives_offset;   // leaf
        int second_child_offset; // interior
    };
    u16 n_primitives; // 0 -> interior node
    u8 axis;          // interior node: xyz x = 0, y = 1, z = 2
};

// BVHBuildNode Definition
struct BVHBuildNode {
    // BVHBuildNode Public Methods
    void init_leaf(int first, int n, const AABB &b) {
        first_prim_offset = first;
        n_primitives = n;
        aabb = b;
        children[0] = children[1] = nullptr;
    }

    void init_interior(int axis, BVHBuildNode *c0, BVHBuildNode *c1) {
        children[0] = c0;
        children[1] = c1;
        aabb = c0->aabb.union_aabb(c1->aabb);
        split_axis = axis;
        n_primitives = 0;
    }

    AABB aabb;
    BVHBuildNode *children[2];
    int split_axis, first_prim_offset, n_primitives;
};

// BVHPrimitive Definition
struct BVHPrimitive {
    BVHPrimitive() {}
    BVHPrimitive(size_t primitive_index, const AABB &bounds)
        : primitiveIndex(primitive_index), aabb(bounds) {}
    size_t primitiveIndex;
    AABB aabb;
    // BVHPrimitive Public Methods
    vec3 centroid() const { return 0.5f * aabb.min + 0.5f * aabb.max; }
};

class BVH {
public:
    BVH() = default;
    BVH(SharedVector<Triangle> *primitives, int max_prims_in_node);
    __device__ cuda::std::optional<Intersection> intersect(Ray &ray, f32 tmax);

private:
    int flattenBVH(BVHBuildNode *node, int *offset);
    BVHBuildNode *buildRecursive(std::span<BVHPrimitive> bvh_primitives, int *total_nodes,
                                 int *ordered_prims_offset,
                                 SharedVector<Triangle> &ordered_prims);

    SharedVector<LinearBVHNode> nodes;

    SharedVector<Triangle> *primitives;
    int max_primitives_in_nodes;
};

#endif // PT_BVH_H
