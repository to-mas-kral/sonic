#ifndef PT_RENDER_CONTEXT_COMMON_H
#define PT_RENDER_CONTEXT_COMMON_H

#include <span>

#include <cuda/std/atomic>
#include <cuda/std/optional>

class Mesh;
class Triangle;
class RenderContext;

struct Intersection;

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

// I can't figure out how to split these into their own headers without running
// into circular include hell...

struct SceneAttribs {
    u32 resx{};
    u32 resy{};
    f32 fov{};
    mat4 camera_to_world = mat4(1.f);
};

struct MeshParams {
    SharedVector<u32> *indices;
    SharedVector<vec3> *pos;
    SharedVector<vec3> *normals = nullptr; // may be null
    SharedVector<vec2> *uvs = nullptr;     // may be null
    u32 material_id;
    cuda::std::optional<u32> light_id = cuda::std::nullopt;
};

class Mesh {
public:
    Mesh(u32 indices_index, u32 pos_index, u32 material_id,
         cuda::std::optional<u32> light_id, RenderContext *rc, u32 num_indices,
         u32 num_vertices, cuda::std::optional<u32> normals_index,
         cuda::std::optional<u32> uvs_index)
        : indices_index(indices_index), pos_index(pos_index), material_id(material_id),
          light_id(light_id), rc(rc), num_indices(num_indices),
          num_vertices(num_vertices), normals_index(normals_index), uvs_index(uvs_index) {
    }

    __host__ __device__ const u32 *get_indices() const;
    __host__ __device__ const vec3 *get_pos() const;

    // Watch out for memory consumption...
    RenderContext *rc;

    u32 indices_index;
    u32 num_indices;
    u32 pos_index;
    u32 num_vertices;
    cuda::std::optional<u32> normals_index;
    cuda::std::optional<u32> uvs_index;

    u32 material_id;
    cuda::std::optional<u32> light_id;
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

class Triangle {
public:
    Triangle(u32 id, u32 mesh_id) : id(id), mesh_id(mesh_id) {}

    __host__ __device__ cuda::std::tuple<u32, u32, u32> get_indices();
    __host__ __device__ cuda::std::tuple<vec3, vec3, vec3> get_pos();

    void set_mesh(Mesh *mesh);
    __device__ cuda::std::optional<Intersection> intersect(Ray &ray);

    __host__ AABB aabb();

    u32 get_id() const { return id; }

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

    __host__ void add_mesh(MeshParams mp);
    __host__ u32 add_material(Material &&material);
    __host__ u32 add_light(Light &&light);
    __host__ u32 add_texture(Texture &&texture);
    __host__ void set_envmap(Envmap &&a_envmap) {
        envmap = std::move(a_envmap);
        has_envmap = true;
    };

    __host__ __device__ const Envmap *get_envmap() { return &envmap; };

    __host__ void fixup_geometry_pointers();
    __host__ void make_acceleration_structure();
    __device__ cuda::std::optional<Intersection> intersect_scene(Ray &ray);

    __host__ __device__ const SharedVector<Material> &get_materials() const {
        return materials;
    }
    __host__ __device__ const SharedVector<Mesh> &get_meshes() const { return meshes; }
    __host__ __device__ const SharedVector<Texture> &get_textures() const {
        return textures;
    }
    __host__ __device__ const SharedVector<Light> &get_lights() const { return lights; }
    __host__ __device__ const SharedVector<u32> &get_indices() const { return indices; }
    __host__ __device__ const SharedVector<vec3> &get_pos() const { return pos; }
    __host__ __device__ const SharedVector<vec3> &get_normals() const { return normals; }
    __host__ __device__ const SharedVector<vec2> &get_uvs() const { return uvs; }
    __host__ __device__ Camera &get_cam() { return cam; }
    __host__ __device__ Framebuffer &get_framebuffer() { return fb; }
    __host__ __device__ u32 get_num_samples() const { return num_samples; }
    __host__ __device__ dim3 get_threads_dim() const { return THREADS_DIM; }
    __host__ __device__ dim3 get_blocks_dim() const { return blocks_dim; }
    __host__ __device__ __forceinline__ const SceneAttribs &get_attribs() const {
        return attribs;
    }

    cuda::atomic<u32> ray_counter{0};
    bool has_envmap = false;

private:
    SharedVector<Mesh> meshes = SharedVector<Mesh>();
    SharedVector<Triangle> triangles = SharedVector<Triangle>();
    SharedVector<Texture> textures = SharedVector<Texture>();

    SharedVector<u32> indices = SharedVector<u32>();
    /// Vertex positions
    SharedVector<vec3> pos = SharedVector<vec3>();
    SharedVector<vec3> normals = SharedVector<vec3>();
    SharedVector<vec2> uvs = SharedVector<vec2>();

    BVH bvh;

    SharedVector<Material> materials = SharedVector<Material>();
    SharedVector<Light> lights = SharedVector<Light>();

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

#endif // PT_RENDER_CONTEXT_COMMON_H
