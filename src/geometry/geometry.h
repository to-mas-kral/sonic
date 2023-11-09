#ifndef PT_GEOMETRY_H
#define PT_GEOMETRY_H

#include <cuda/std/array>
#include <cuda/std/optional>

#include "../utils/numtypes.h"
#include "../utils/shared_vector.h"
#include "aabb.h"
#include "intersection.h"

struct MeshParams {
    SharedVector<u32> *indices;
    SharedVector<vec3> *pos;
    SharedVector<vec3> *normals = nullptr; // may be null
    SharedVector<vec2> *uvs = nullptr;     // may be null
    u32 material_id;
    cuda::std::optional<u32> light_id = cuda::std::nullopt;
};

struct TriangleMeshes;
struct Intersection;

struct Mesh {
    Mesh(u32 indices_index, u32 pos_index, u32 material_id,
         cuda::std::optional<u32> p_light_id, TriangleMeshes *tm, u32 num_indices,
         u32 num_vertices, cuda::std::optional<u32> p_normals_index,
         cuda::std::optional<u32> p_uvs_index)
        : indices_index(indices_index), pos_index(pos_index), material_id(material_id),
          tm(tm), num_indices(num_indices), num_vertices(num_vertices) {

        if (p_light_id.has_value()) {
            light_id = p_light_id.value();
            has_light = true;
        }

        if (p_normals_index.has_value()) {
            normals_index = p_normals_index.value();
            has_normals = true;
        }

        if (p_uvs_index.has_value()) {
            uvs_index = p_uvs_index.value();
            has_uvs = true;
        }
    }

    __host__ __device__ const u32 *get_indices() const;
    __host__ __device__ const vec3 *get_pos() const;

    // What's exctually loaded by OptiX PT
    // Don't use optional<T> due too much memory consumption
    bool has_normals = false;
    bool has_uvs = false;
    bool has_light = false;
    bool padding; // for 128-bit load
    u32 padding2;
    u32 light_id;
    u32 material_id;

    // What's needed on the CPU...
    TriangleMeshes *tm;
    u32 uvs_index = 0xdeadbeef;
    u32 indices_index = 0xdeadbeef;
    u32 pos_index;
    u32 normals_index;
    u32 num_vertices;
    u32 num_indices;
};

struct Triangle {
    Triangle(u32 id, u32 mesh_id) : id(id), mesh_id(mesh_id) {}

    __host__ __device__ cuda::std::array<u32, 3> get_indices();
    __host__ __device__ cuda::std::array<vec3, 3> get_pos();

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

// OptiX requires SOA layout
struct TriangleMeshes {
    SharedVector<Mesh> meshes = SharedVector<Mesh>();
    SharedVector<Triangle> triangles = SharedVector<Triangle>();

    SharedVector<u32> indices = SharedVector<u32>();
    SharedVector<vec3> pos = SharedVector<vec3>();
    SharedVector<vec3> normals = SharedVector<vec3>();
    SharedVector<vec2> uvs = SharedVector<vec2>();
};

struct SphereParams {
    vec3 center;
    f32 radius;
    u32 material_id;
    cuda::std::optional<u32> light_id;
};

// OptiX requires SOA layout
struct Spheres {
    SharedVector<vec3> centers = SharedVector<vec3>();
    SharedVector<f32> radiuses = SharedVector<f32>();
    SharedVector<u32> material_ids = SharedVector<u32>();
    SharedVector<bool> has_light = SharedVector<bool>();
    SharedVector<u32> light_ids = SharedVector<u32>();
    u32 num_spheres = 0;
};

struct Geometry {
    TriangleMeshes meshes{};
    Spheres spheres{};

    __host__ void add_mesh(MeshParams mp);
    __host__ void add_sphere(SphereParams sp);
    __host__ void fixup_geometry_pointers();
};

#endif // PT_GEOMETRY_H
