#ifndef PT_RENDER_CONTEXT_COMMON_H
#define PT_RENDER_CONTEXT_COMMON_H

#ifndef PT_RENDER_CONTEXT_H
#define PT_RENDER_CONTEXT_H

#include <span>

#include <cuda/std/atomic>

class Mesh;
class Triangle;
class RenderContext;

#include "camera.h"
#include "envmap.h"
#include "framebuffer.h"
#include "geometry/aabb.h"
#include "geometry/axis.h"
#include "geometry/intersection.h"
#include "geometry/ray.h"
#include "light.h"
#include "material.h"
#include "utils/numtypes.h"
#include "utils/shared_vector.h"

// TODO: I can't figure out how to split these into their own headers without running
// into circular include hell...

struct SceneAttribs {
    u32 resx{};
    u32 resy{};
    f32 fov{};
    mat4 camera_to_world = mat4(1.f);
};

class Mesh {
public:
    Mesh(u32 indices_index, u32 pos_index, u32 material_id, i32 light_id,
         RenderContext *rc)
        : indices_index(indices_index), pos_index(pos_index), material_id(material_id),
          light_id(light_id), rc(rc) {}
    /*Mesh(SharedVector<u32> &&a_indices, SharedVector<vec3> &&a_pos, u32 materialId,
         i32 light_id = -1)
        : indices(std::move(a_indices)), pos(std::move(a_pos)), material_id(materialId),
          light_id(light_id) {
        assert(indices.len() % 3 == 0);
    }

    Mesh(SharedVector<u32> &&a_indices, SharedVector<f32> &&a_pos, u32 a_material_id,
         i32 light_id = -1)
        : indices(std::move(a_indices)), material_id(a_material_id), light_id(light_id) {
        assert(indices.len() % 3 == 0);
        assert(a_pos.len() % 3 == 0);

        pos = SharedVector<vec3>(a_pos.len() / 3);
        for (int i = 0; i < a_pos.len(); i += 3) {
            auto p = vec3(a_pos[i], a_pos[i + 1], a_pos[i + 2]);
            pos.push(std::move(p));
        }
    }*/

    // Mesh(Mesh const &) = delete;

    // Mesh &operator=(Mesh const &) = delete;

    Mesh(Mesh &&other) noexcept {
        /*indices = std::move(other.indices);
        pos = std::move(other.pos);*/
        rc = other.rc;
        indices_index = other.indices_index;
        pos_index = other.pos_index;
        material_id = other.material_id;
        light_id = other.light_id;
    };

    Mesh &operator=(Mesh &&other) noexcept {
        /*indices = std::move(other.indices);
        pos = std::move(other.pos);*/
        rc = other.rc;
        indices_index = other.indices_index;
        pos_index = other.pos_index;
        material_id = other.material_id;
        light_id = other.light_id;

        return *this;
    };

    __host__ __device__ const u32 *get_indices() const;
    __host__ __device__ const vec3 *get_pos() const;
    __device__ u32 get_material_id() const { return material_id; }
    // TODO: use optional
    __device__ bool has_light() const { return light_id >= 0; }
    __device__ i32 get_light_id() const { return light_id; }

private:
    // Indices into the global array in render_context...
    RenderContext *rc;
    u32 indices_index;
    u32 pos_index;
    /*SharedVector<u32> indices;
    /// Vertices positions
    SharedVector<vec3> pos;*/
    u32 material_id;
    // TODO: use optional when available
    // Negative means no light
    i32 light_id;
};

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
    // TODO: use optional when available
    __device__ bool intersect(Intersection &its, Ray &ray, f32 tmax);

private:
    int flattenBVH(BVHBuildNode *node, int *offset);
    BVHBuildNode *buildRecursive(std::span<BVHPrimitive> bvh_primitives, int *total_nodes,
                                 int *ordered_prims_offset,
                                 SharedVector<Triangle> &ordered_prims);

    SharedVector<LinearBVHNode> nodes;

    SharedVector<Triangle> *primitives;
    int max_primitives_in_nodes;
};

class Triangle {
public:
    Triangle(u32 id, u32 mesh_id) : id(id), mesh_id(mesh_id) {}

    __host__ __device__ cuda::std::tuple<u32, u32, u32> get_indices();
    __host__ __device__ cuda::std::tuple<vec3, vec3, vec3> get_pos();

    u32 get_mesh_id() { return mesh_id; };
    void set_mesh(Mesh *mesh);
    // TODO: use cuda::std::optional when available
    /// Möller-Trumbore intersection algorithm
    __device__ bool intersect(Intersection &its, Ray &ray);

    __host__ AABB aabb();

    u32 get_id() const { return id; }

private:
    union {
        Mesh *mesh;
        u32 mesh_id;
    };
    /// Triangle id, unmultiplied.
    u32 id;
};

/// Render Context is a collection of data needed for the render kernels to to their
/// job.
class RenderContext {
public:
    explicit RenderContext(u32 num_samples, SceneAttribs &attribs);

    __host__ void add_mesh(SharedVector<u32> &&indices, SharedVector<vec3> &&pos,
                           u32 material_id, i32 light_id);
    __host__ u32 add_material(Material &&material);
    __host__ u32 add_light(Light &&light);
    __host__ void set_envmap(Envmap &a_envmap) { envmap = std::move(a_envmap); };
    // TODO: use an optional when available
    __host__ __device__ const Envmap *get_envmap() { return &envmap; };

    __host__ void make_acceleration_structure();
    __device__ bool intersect_scene(Intersection &its, Ray &ray);

    __host__ __device__ const SharedVector<Material> &get_materials() const {
        return materials;
    }
    __host__ __device__ const SharedVector<Mesh> &get_meshes() const { return meshes; }
    __host__ __device__ const SharedVector<Light> &get_lights() const { return lights; }
    __host__ __device__ const SharedVector<u32> &get_indices() const { return indices; }
    __host__ __device__ const SharedVector<vec3> &get_pos() const { return pos; }
    __host__ __device__ Camera &get_cam() { return cam; }
    __host__ __device__ Framebuffer &get_framebuffer() { return fb; }
    __host__ __device__ u32 get_num_samples() const { return num_samples; }
    __host__ __device__ dim3 get_threads_dim() const { return THREADS_DIM; }
    __host__ __device__ dim3 get_blocks_dim() const { return blocks_dim; }
    __host__ __device__ __forceinline__ const SceneAttribs &get_attribs() const {
        return attribs;
    }

    cuda::atomic<u32> ray_counter{0};

private:
    SharedVector<Mesh> meshes;
    SharedVector<Triangle> triangles; // It's a triangle vector, but could be a general
                                      // shape in the future...

    // TODO: Vertex data is duplicated for simplicity for now (for OptiX)...
    SharedVector<u32> indices;
    /// Vertex positions
    SharedVector<vec3> pos;

    BVH bvh;

    SharedVector<Material> materials;
    SharedVector<Light> lights;

    Envmap envmap;

    Camera cam;
    Framebuffer fb;

    SceneAttribs attribs;

    u32 image_x;
    u32 image_y;
    u32 num_samples;

    const u32 THREADS_DIM_SIZE = 8;
    const dim3 THREADS_DIM = dim3(THREADS_DIM_SIZE, THREADS_DIM_SIZE);
    dim3 blocks_dim;
};

#endif // PT_RENDER_CONTEXT_H

#endif // PT_RENDER_CONTEXT_COMMON_H
