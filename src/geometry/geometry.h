#ifndef PT_GEOMETRY_H
#define PT_GEOMETRY_H

#include <cuda/std/array>
#include <cuda/std/optional>
#include <cuda/std/utility>

#include "../emitter.h"
#include "../math/sampling.h"
#include "../utils/numtypes.h"
#include "../utils/shared_vector.h"

enum class ShapeType : u8 {
    Mesh = 0,
    Sphere = 1,
};

// TODO: optimization - could merge triangle index into type...
struct ShapeIndex {
    ShapeType type;
    u32 index;
    u32 triangle_index;
};

struct ShapeSample {
    vec3 pos;
    vec3 normal;
    f32 pdf;
    f32 area;
};

// TODO: could apply SOA layout to this astruct as well...
struct Mesh {
    Mesh(u32 indices_index, u32 pos_index, u32 material_id,
         cuda::std::optional<u32> p_lights_start_id, u32 num_indices, u32 num_vertices,
         cuda::std::optional<u32> p_normals_index, cuda::std::optional<u32> p_uvs_index);

    __host__ __device__ __forceinline__ u32 num_triangles() const {
        return num_indices / 3;
    };

    // What's exctually loaded by OptiX PT
    // Don't use optional<T> due too much memory consumption
    bool has_normals = false;
    bool has_uvs = false;
    bool has_light = false;
    u32 lights_start_id;
    u32 material_id;

    // What's needed on the CPU...
    u32 uvs_index = 0xdeadbeef;
    u32 indices_index = 0xdeadbeef;
    u32 pos_index;
    u32 normals_index;
    u32 num_vertices;
    u32 num_indices;
};

// Used only for mesh creation
struct MeshParams {
    SharedVector<u32> *indices;
    SharedVector<vec3> *pos;
    SharedVector<vec3> *normals = nullptr; // may be null
    SharedVector<vec2> *uvs = nullptr;     // may be null
    u32 material_id;
    cuda::std::optional<Emitter> emitter = cuda::std::nullopt;
};

// OptiX requires SOA layout
struct Meshes {
    SharedVector<Mesh> meshes = SharedVector<Mesh>();

    SharedVector<u32> indices = SharedVector<u32>();
    SharedVector<vec3> pos = SharedVector<vec3>();
    SharedVector<vec3> normals = SharedVector<vec3>();
    SharedVector<vec2> uvs = SharedVector<vec2>();

    __host__ __device__ __forceinline__ uvec3 get_tri_indices(u32 mesh_indices_index,
                                                              u32 triangle) const {
        u32 index = triangle * 3;
        u32 i0 = indices[mesh_indices_index + index];
        u32 i1 = indices[mesh_indices_index + index + 1];
        u32 i2 = indices[mesh_indices_index + index + 2];
        return uvec3(i0, i1, i2);
    };

    __host__ __device__ __forceinline__ cuda::std::array<vec3, 3>
    get_tri_pos(u32 mesh_pos_index, const uvec3 &tri_indices) const {
        vec3 p0 = pos[mesh_pos_index + tri_indices.x];
        vec3 p1 = pos[mesh_pos_index + tri_indices.y];
        vec3 p2 = pos[mesh_pos_index + tri_indices.z];
        return {p0, p1, p2};
    };

    __host__ __device__ __forceinline__ f32 calc_tri_area(u32 mesh_indices_index,
                                                          u32 mesh_pos_index,
                                                          u32 triangle) const {
        uvec3 tri_indices = get_tri_indices(mesh_indices_index, triangle);
        const auto [p0, p1, p2] = get_tri_pos(mesh_pos_index, tri_indices);
        vec3 v1 = p1 - p0;
        vec3 v2 = p2 - p0;
        vec3 cross = glm::cross(v1, v2);
        return glm::length(cross) / 2.f;
    };

    __device__ __forceinline__ static vec3 calc_normal(bool has_normals, u32 i0, u32 i1,
                                                       u32 i2, const vec3 *normals,
                                                       const vec3 &bar, const vec3 &p0,
                                                       const vec3 &p1, const vec3 &p2) {
        if (has_normals) {
            vec3 n0 = normals[i0];
            vec3 n1 = normals[i1];
            vec3 n2 = normals[i2];
            return glm::normalize(barycentric_interp(bar, n0, n1, n2));
        } else {
            vec3 v0 = p1 - p0;
            vec3 v1 = p2 - p0;
            vec3 normal = glm::normalize(cross(v0, v1));
            if (glm::any(glm::isnan(normal))) {
                // TODO: Degenerate triangle hack...
                normal = glm::normalize(vec3(0.5f, 0.3f, -0.7f));
            }

            return normal;
        }
    }

    __device__ __forceinline__ static vec2 calc_uvs(bool has_uvs, u32 i0, u32 i1, u32 i2,
                                                    const vec2 *uvs, const vec3 &bar) {
        // Idk what's suppossed to happen here without explicit UVs..
        vec2 uv = vec2(0.);
        if (has_uvs) {
            vec2 uv0 = uvs[i0];
            vec2 uv1 = uvs[i1];
            vec2 uv2 = uvs[i2];
            uv = barycentric_interp(bar, uv0, uv1, uv2);
        }

        return uv;
    }

    __device__ __forceinline__ ShapeSample sample(ShapeIndex si,
                                                  const vec3 &sample) const {
        auto &mesh = meshes[si.index];

        const vec3 bar = sample_uniform_triangle(vec2(sample.y, sample.z));
        uvec3 tri_indices = get_tri_indices(mesh.indices_index, si.triangle_index);
        const auto tri_pos = get_tri_pos(mesh.pos_index, tri_indices);
        vec3 sampled_pos = barycentric_interp(bar, tri_pos[0], tri_pos[1], tri_pos[2]);

        vec3 normal =
            calc_normal(mesh.has_normals, tri_indices.x, tri_indices.y, tri_indices.z,
                        normals.get_ptr(), bar, tri_pos[0], tri_pos[1], tri_pos[2]);

        f32 area = calc_tri_area(mesh.indices_index, mesh.pos_index, si.triangle_index);

        return ShapeSample{
            .pos = sampled_pos,
            .normal = normal,
            .pdf = 1.f / area,
            .area = area,
        };
    }
};

// Used only for sphere creation
struct SphereParams {
    vec3 center;
    f32 radius;
    u32 material_id;
    cuda::std::optional<Emitter> emitter = cuda::std::nullopt;
};

// OptiX requires SOA layout
struct Spheres {
    SharedVector<vec3> centers = SharedVector<vec3>();
    SharedVector<f32> radiuses = SharedVector<f32>();
    SharedVector<u32> material_ids = SharedVector<u32>();
    SharedVector<bool> has_light = SharedVector<bool>();
    SharedVector<u32> light_ids = SharedVector<u32>();
    u32 num_spheres = 0;

    __device__ __forceinline__ ShapeSample sample(u32 index, const vec3 &illuminated_pos,
                                                  const vec3 &sample) const {
        vec3 sample_dir = sample_uniform_sphere(sample);
        vec3 center = centers[index];
        f32 radius = radiuses[index];

        vec3 pos = center + radius * sample_dir;

        return ShapeSample{
            .pos = pos,
            .normal = calc_normal(pos, center),
            .pdf = UNIFORM_SPHERE_SAMPLE_PDF,
            .area = calc_sphere_area(radius),
        };
    }

    __device__ __forceinline__ static f32 calc_sphere_area(f32 radius) {
        return 4.f * M_PIf * sqr(radius);
    }

    __host__ __device__ f32 calc_sphere_area(u32 sphere_id) const {
        f32 radius = radiuses[sphere_id];
        return 4.f * M_PIf * sqr(radius);
    }

    __device__ __forceinline__ static vec3 calc_normal(const vec3 &pos,
                                                       const vec3 &center) {
        return glm::normalize(pos - center);
    }

    __device__ __forceinline__ static vec2 calc_uvs(const vec3 &normal) {
        // TODO: Sphere UV mapping could be wrong, test...
        // (1 / 2pi, 1 / pi)
        const vec2 pi_reciprocals = vec2(0.1591f, 0.3183f);
        vec2 uv = vec2(atan2(-normal.z, -normal.x), asin(normal.y));
        uv *= pi_reciprocals;
        uv += 0.5;
        return uv;
    }
};

struct Geometry {
    Meshes meshes{};
    Spheres spheres{};

    __host__ void add_mesh(const MeshParams& mp, cuda::std::optional<u32> lights_start_id);
    __host__ void add_sphere(SphereParams sp, cuda::std::optional<u32> light_id);

    /// Based on the shape type, returns the  index of the *next* shape in that category.
    __host__ u32 get_next_shape_index(ShapeType type) const {
        switch (type) {
        case ShapeType::Mesh:
            return meshes.meshes.size();
        case ShapeType::Sphere:
            return spheres.num_spheres;
        default:
            assert(false);
        }
    }

    __device__ __forceinline__ ShapeSample sample_shape(ShapeIndex si, const vec3 &pos,
                                                        const vec3 &sample) const {
        switch (si.type) {
        case ShapeType::Mesh:
            return meshes.sample(si, sample);
        case ShapeType::Sphere:
            return spheres.sample(si.index, pos, sample);
        default:
            assert(false);
        }
    }

    __host__ __device__ __forceinline__ f32 shape_area(ShapeIndex si) const {
        switch (si.type) {
        case ShapeType::Mesh: {
            auto &mesh = meshes.meshes[si.index];
            return meshes.calc_tri_area(mesh.indices_index, mesh.pos_index,
                                        si.triangle_index);
        }
        case ShapeType::Sphere:
            return spheres.calc_sphere_area(si.index);
        default:
            assert(false);
        }
    }
};

#endif // PT_GEOMETRY_H
