#ifndef PT_GEOMETRY_H
#define PT_GEOMETRY_H

#include "../materials/material_id.h"
#include "../math/vecmath.h"
#include "../scene/emitter.h"
#include "../scene/texture.h"
#include "../utils/basic_types.h"

#include <vector>

enum class ShapeType : u8 {
    Mesh = 0,
    Sphere = 1,
};

// OPTIMIZE: could merge triangle index into type...
struct ShapeIndex {
    ShapeType type;
    u32 index;
    u32 triangle_index;
};

struct ShapeLightSample {
    point3 pos;
    norm_vec3 normal{1.f, 0.f, 0.f};
    f32 pdf;
    spectral emission;
};

struct MeshParams;

struct Mesh {
    Mesh(const MeshParams &mp, Option<u32> p_lights_start_id);

    u32
    num_triangles() const;

    uvec3
    get_tri_indices(u32 triangle) const;

    std::array<point3, 3>
    get_tri_pos(const uvec3 &tri_indices) const;

    f32
    tri_area(u32 triangle) const;

    norm_vec3
    calc_normal(u32 triangle, vec3 bar, bool want_geometric_normal) const;

    vec2
    calc_uvs(u32 triangle_index, const vec3 &bar) const;

    Mesh(const Mesh &other) = delete;

    Mesh(Mesh &&other) noexcept
        : num_verts(other.num_verts), num_indices(other.num_indices), pos(other.pos),
          normals(other.normals), uvs(other.uvs), indices(other.indices),
          alpha{other.alpha}, has_light(other.has_light),
          lights_start_id(other.lights_start_id),
          material_id(std::move(other.material_id)) {
        other.pos = nullptr;
        other.normals = nullptr;
        other.uvs = nullptr;
        other.indices = nullptr;
    }

    Mesh &
    operator=(const Mesh &other) = delete;

    Mesh &
    operator=(Mesh &&other) noexcept {
        if (this == &other)
            return *this;
        num_verts = other.num_verts;
        num_indices = other.num_indices;
        pos = other.pos;
        normals = other.normals;
        alpha = other.alpha;
        uvs = other.uvs;
        indices = other.indices;
        has_light = other.has_light;
        lights_start_id = other.lights_start_id;
        material_id = std::move(other.material_id);

        other.pos = nullptr;
        other.normals = nullptr;
        other.uvs = nullptr;
        other.indices = nullptr;

        return *this;
    }

    ~
    Mesh() {
        if (pos) {
            std::free(pos);
        }

        if (normals) {
            std::free(normals);
        }

        if (uvs) {
            std::free(uvs);
        }

        if (indices) {
            std::free(indices);
        }
    }

    u32 num_verts;
    u32 num_indices;
    point3 *pos{nullptr};
    vec3 *normals{nullptr};
    vec2 *uvs{nullptr};
    u32 *indices{nullptr};
    FloatTexture *alpha{nullptr};

    bool has_light = false;
    u32 lights_start_id;
    MaterialId material_id;
};

// TODO: also think about validation... validating the index buffers would be robust
// against UB...
// TODO: refactor to norm_vec3... ?
// Used only for mesh creation
struct MeshParams {
    u32 *indices;
    u32 num_indices;
    point3 *pos;
    vec3 *normals = nullptr; // may be null
    vec2 *uvs = nullptr;     // may be null
    u32 num_verts;
    MaterialId material_id;
    Option<Emitter> emitter{};
    FloatTexture *alpha{nullptr};
};

// SOA layout
struct Meshes {
    std::vector<Mesh> meshes{};

    ShapeLightSample
    sample(ShapeIndex si, const vec3 &sample) const;
};

// Used only for sphere creation
struct SphereParams {
    point3 center;
    f32 radius;
    MaterialId material_id;
    Option<Emitter> emitter = {};
    FloatTexture *alpha{nullptr};
};

struct SphereVertex {
    point3 pos;
    f32 radius;
};

// SOA layout
struct Spheres {
    std::vector<SphereVertex> vertices{};
    std::vector<MaterialId> material_ids{};
    std::vector<bool> has_light{};
    std::vector<u32> light_ids{};
    std::vector<FloatTexture *> alphas{};
    u32 num_spheres = 0;

    ShapeLightSample
    sample(u32 index, const point3 &illuminated_pos, const vec3 &sample) const;

    static f32
    calc_sphere_area(f32 radius);

    f32
    calc_sphere_area(u32 sphere_id) const;

    static norm_vec3
    calc_normal(const point3 &pos, const point3 &center,
                bool want_geometric_normal = false);

    static vec2
    calc_uvs(const vec3 &normal);
};

struct InstancedObj {
    Meshes meshes{};
    Spheres spheres{};
};

/// InstancedObj is one intanced object (let's say a tree).
/// Then the instances themselves are stored in a SOA layout, so that it can be shared
/// with Embree.
/// 'indices' maps from the instances themselves to the instanced objects.
struct Instances {
    std::vector<InstancedObj> instanced_objs{};
    std::vector<SquareMatrix4> world_from_instances{};
    std::vector<SquareMatrix4> wfi_inv_trans{};
    std::vector<u32> indices{};
};

struct Geometry {
    Meshes meshes{};
    Spheres spheres{};

    Instances instances{};

    void
    add_mesh(const MeshParams &mp, Option<u32> lights_start_id);
    void
    add_sphere(SphereParams sp, Option<u32> light_id);

    /// Based on the shape type, returns the  index of the *next* shape in that category.
    u32
    get_next_shape_index(ShapeType type) const;

    ShapeLightSample
    sample_shape(ShapeIndex si, const point3 &pos, const vec3 &sample) const;

    f32
    shape_area(ShapeIndex si) const;
};

#endif // PT_GEOMETRY_H
