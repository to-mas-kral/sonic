#include <algorithm>
#include <vector>

#include <cuda/std/array>

#include "render_context_common.h"

/*
 * EVERYTHING here was taken from PBRTv4 !
 * */

// BVHSplitBucket Definition
struct BVHSplitBucket {
    int count = 0;
    AABB bounds;
};

int BVH::flattenBVH(BVHBuildNode *node, int *offset) {
    LinearBVHNode *linear_node = &nodes[*offset];
    linear_node->aabb = node->aabb;
    int node_offset = (*offset)++;
    if (node->n_primitives > 0) {
        assert(!node->children[0] && !node->children[1]);
        assert(node->n_primitives < 65536);
        linear_node->primitives_offset = node->first_prim_offset;
        linear_node->n_primitives = node->n_primitives;
    } else {
        // Create interior flattened BVH node
        linear_node->axis = node->split_axis;
        linear_node->n_primitives = 0;
        flattenBVH(node->children[0], offset);
        linear_node->second_child_offset = flattenBVH(node->children[1], offset);
    }
    return node_offset;
}

// BVHAggregate Method Definitions
BVH::BVH(SharedVector<Triangle> *primitives, int max_primitives_in_nodes)
    : primitives(primitives), max_primitives_in_nodes(max_primitives_in_nodes) {
    assert(primitives->len() > 0);

    // Build BVH from primitives
    // Initialize bvhPrimitives array for primitives
    std::vector<BVHPrimitive> bvh_primitives(primitives->len());
    for (size_t i = 0; i < primitives->len(); ++i)
        bvh_primitives[i] = BVHPrimitive(i, (*primitives)[i].aabb());

    SharedVector<Triangle> ordered_prims(primitives->len());
    ordered_prims.assume_all_init();

    BVHBuildNode *root;

    int total_nodes = 0;
    int ordered_prims_offset = 0;

    root = buildRecursive(std::span<BVHPrimitive>(bvh_primitives), &total_nodes,
                          &ordered_prims_offset, ordered_prims);

    assert(ordered_prims_offset == ordered_prims.len());

    primitives->swap(&ordered_prims);
    bvh_primitives.resize(0);

    nodes = SharedVector<LinearBVHNode>(total_nodes);
    nodes.assume_all_init();
    int offset = 0;
    flattenBVH(root, &offset);
    assert(total_nodes == offset);
}

BVHBuildNode *BVH::buildRecursive(std::span<BVHPrimitive> bvh_primitives,
                                  int *total_nodes, int *ordered_prims_offset,
                                  SharedVector<Triangle> &ordered_prims) {

    auto *node = new BVHBuildNode();

    // Initialize _BVHBuildNode_ for primitive range
    ++*total_nodes;
    // Compute aabb of all primitives in BVH node
    AABB bounds;
    for (const auto &prim : bvh_primitives)
        bounds = bounds.union_aabb(prim.aabb);

    if (bounds.area() == 0 || bvh_primitives.size() == 1) {
        // Create leaf _BVHBuildNode_
        int first_prim_offset = *ordered_prims_offset;
        *ordered_prims_offset += bvh_primitives.size();

        for (size_t i = 0; i < bvh_primitives.size(); ++i) {
            int index = bvh_primitives[i].primitiveIndex;
            ordered_prims[first_prim_offset + i] = (*primitives)[index];
        }
        node->init_leaf(first_prim_offset, bvh_primitives.size(), bounds);
        return node;

    } else {
        // Compute bound of primitive centroids and choose split dimension _dim_
        AABB centroid_bounds;
        for (const auto &prim : bvh_primitives)
            centroid_bounds = centroid_bounds.union_point(prim.centroid());
        int dim = centroid_bounds.max_axis();

        // Partition primitives into two sets and build children
        if (centroid_bounds.max[dim] == centroid_bounds.min[dim]) {
            // Create leaf _BVHBuildNode_
            int firstPrimOffset = *ordered_prims_offset;
            *ordered_prims_offset += bvh_primitives.size();

            for (size_t i = 0; i < bvh_primitives.size(); ++i) {
                int index = bvh_primitives[i].primitiveIndex;
                ordered_prims[firstPrimOffset + i] = (*primitives)[index];
            }
            node->init_leaf(firstPrimOffset, bvh_primitives.size(), bounds);
            return node;

        } else {
            int mid = bvh_primitives.size() / 2;
            // Partition primitives based on _splitMethod_

            int split_method = 1;

            switch (split_method) {
            case 0: {
                // Partition primitives into equally sized subsets
                mid = bvh_primitives.size() / 2;
                std::nth_element(bvh_primitives.begin(), bvh_primitives.begin() + mid,
                                 bvh_primitives.end(),
                                 [dim](const BVHPrimitive &a, const BVHPrimitive &b) {
                                     return a.centroid()[dim] < b.centroid()[dim];
                                 });

                break;
            }
            case 1:
            default: {
                // Partition primitives using approximate SAH
                if (bvh_primitives.size() <= 2) {
                    // Partition primitives into equally sized subsets
                    mid = bvh_primitives.size() / 2;
                    std::nth_element(bvh_primitives.begin(), bvh_primitives.begin() + mid,
                                     bvh_primitives.end(),
                                     [dim](const BVHPrimitive &a, const BVHPrimitive &b) {
                                         return a.centroid()[dim] < b.centroid()[dim];
                                     });

                } else {
                    // Allocate _BVHSplitBucket_ for SAH partition buckets
                    constexpr int n_buckets = 12;
                    BVHSplitBucket buckets[n_buckets];

                    // Initialize _BVHSplitBucket_ for SAH partition buckets
                    for (const auto &prim : bvh_primitives) {
                        int b = (f32)n_buckets *
                                centroid_bounds.offset_of(prim.centroid())[dim];
                        if (b == n_buckets)
                            b = n_buckets - 1;
                        assert(b >= 0);
                        assert(b < n_buckets);
                        buckets[b].count++;
                        buckets[b].bounds = buckets[b].bounds.union_aabb(prim.aabb);
                    }

                    // Compute costs for splitting after each bucket
                    constexpr int n_splits = n_buckets - 1;
                    f32 costs[n_splits] = {};
                    // Partially initialize _costs_ using a forward scan over splits
                    int count_below = 0;
                    AABB bound_below;
                    for (int i = 0; i < n_splits; ++i) {
                        bound_below = bound_below.union_aabb(buckets[i].bounds);
                        count_below += buckets[i].count;
                        costs[i] += count_below * bound_below.area();
                    }

                    // Finish initializing _costs_ using a backward scan over splits
                    int count_above = 0;
                    AABB bound_above;
                    for (int i = n_splits; i >= 1; --i) {
                        bound_above = bound_above.union_aabb(buckets[i].bounds);
                        count_above += buckets[i].count;
                        costs[i - 1] += count_above * bound_above.area();
                    }

                    // Find bucket to split at that minimizes SAH metric
                    int min_cost_split_bucket = -1;
                    f32 min_cost = cuda::std::numeric_limits<f32>::max();
                    for (int i = 0; i < n_splits; ++i) {
                        // Compute cost for candidate split and update minimum if
                        // necessary
                        if (costs[i] < min_cost) {
                            min_cost = costs[i];
                            min_cost_split_bucket = i;
                        }
                    }
                    // Compute leaf cost and SAH split cost for chosen split
                    f32 leaf_cost = bvh_primitives.size();
                    min_cost = 1.f / 2.f + min_cost / bounds.area();

                    // Either create leaf or split primitives at selected SAH bucket
                    if (bvh_primitives.size() > max_primitives_in_nodes ||
                        min_cost < leaf_cost) {
                        auto mid_iter = std::partition(
                            bvh_primitives.begin(), bvh_primitives.end(),
                            [=](const BVHPrimitive &bp) {
                                int b = n_buckets *
                                        centroid_bounds.offset_of(bp.centroid())[dim];
                                if (b == n_buckets)
                                    b = n_buckets - 1;
                                return b <= min_cost_split_bucket;
                            });
                        mid = mid_iter - bvh_primitives.begin();
                    } else {
                        // Create leaf _BVHBuildNode_
                        int first_prim_offset = *ordered_prims_offset;
                        *ordered_prims_offset += bvh_primitives.size();

                        for (size_t i = 0; i < bvh_primitives.size(); ++i) {
                            int index = bvh_primitives[i].primitiveIndex;
                            ordered_prims[first_prim_offset + i] = (*primitives)[index];
                        }
                        node->init_leaf(first_prim_offset, bvh_primitives.size(), bounds);
                        return node;
                    }
                }

                break;
            }
            }

            BVHBuildNode *children[2];

            // Recursively build child BVHs sequentially
            children[0] = buildRecursive(bvh_primitives.subspan(0, mid), total_nodes,
                                         ordered_prims_offset, ordered_prims);
            children[1] = buildRecursive(bvh_primitives.subspan(mid), total_nodes,
                                         ordered_prims_offset, ordered_prims);

            node->init_interior(dim, children[0], children[1]);
        }
    }

    return node;
}

__device__ bool BVH::intersect(Intersection &its, Ray &ray, f32 tmax) {
    vec3 inv_dir = vec3(1.f) / ray.dir;
    bvec3 dir_is_neg = glm::lessThan(inv_dir, vec3(0.f));

    u32 current_node_index = 0;
    u32 to_visit_offset = 0;
    cuda::std::array<u32, 32> nodes_to_visit{};

    bool its_found = false;
    Intersection min_its{};
    min_its.t = cuda::std::numeric_limits<f32>::max();

    while (true) {
        auto node = &nodes[current_node_index];
        if (node->aabb.intersects(ray, tmax, inv_dir, dir_is_neg)) {
            if (node->n_primitives > 0) {
                // Leaf node
                for (int i = 0; i < node->n_primitives; ++i) {
                    // Check for intersection with primitive in BVH node
                    auto prim = (*primitives)[node->primitives_offset + i];

                    Intersection cur_its{};

                    bool does_its = prim.intersect(cur_its, ray);
                    if (does_its && (cur_its.t < min_its.t)) {
                        tmax = cur_its.t;
                        min_its = cur_its;
                        its_found = true;
                    }
                }
                if (to_visit_offset == 0) {
                    break;
                }
                current_node_index = nodes_to_visit[--to_visit_offset];

            } else {
                // Interior node
                // Put further node on stack, advance to near node
                if (dir_is_neg[node->axis]) {
                    nodes_to_visit[to_visit_offset++] = current_node_index + 1;
                    current_node_index = node->second_child_offset;
                } else {
                    nodes_to_visit[to_visit_offset++] = node->second_child_offset;
                    current_node_index = current_node_index + 1;
                }
            }
        } else {
            if (to_visit_offset == 0) {
                break;
            } else {
                current_node_index = nodes_to_visit[--to_visit_offset];
            }
        }
    }

    its = min_its;
    return its_found;
}
