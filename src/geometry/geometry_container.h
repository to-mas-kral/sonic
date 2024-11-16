#ifndef PT_GEOMETRY_CONTAINER_H
#define PT_GEOMETRY_CONTAINER_H

#include "../materials/material_id.h"
#include "../math/vecmath.h"
#include "../scene/emitter.h"
#include "../scene/texture.h"
#include "../utils/basic_types.h"
#include "geometry_storage.h"
#include "instance_id.h"

#include <vector>

enum class ShapeType : u8 {
    Mesh = 0,
    Sphere = 1,
};

struct ShapeIndex {
    ShapeType type;
    u32 index;
    u32 triangle_index;
};

struct ShapeLightSample {
    point3 pos;
    norm_vec3 normal;
    f32 pdf{};
    spectral emission;
};

struct MeshParams;

class Mesh {
public:
    Mesh(const MeshParams &mp, std::optional<u32> p_lights_start_id);

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

    u32
    num_verts() const {
        return m_num_verts;
    }

    u32
    num_indices() const {
        return m_num_indices;
    }

    point3 *
    pos() const {
        return m_pos;
    }

    u32 *
    indices() const {
        return m_indices;
    }

    FloatTexture *
    alpha() const {
        return m_alpha;
    }

    bool
    has_light() const {
        return m_has_light;
    }

    u32
    lights_start_id() const {
        return m_lights_start_id;
    }

    MaterialId
    material_id() const {
        return m_material_id;
    }

private:
#ifdef TEST_PUBLIC
public:
#endif
    u32 m_num_verts;
    u32 m_num_indices;
    point3 *m_pos{nullptr};
    vec3 *m_normals{nullptr};
    vec2 *m_uvs{nullptr};
    u32 *m_indices{nullptr};
    FloatTexture *m_alpha{nullptr};

    bool m_has_light = false;
    u32 m_lights_start_id{0};
    MaterialId m_material_id;
};

// TODO: refactor to norm_vec3... ?
// Used only for mesh creation
struct MeshParams {
    MeshParams(u32 *const indices, const u32 num_indices, point3 *const pos,
               vec3 *const normals, vec2 *const uvs, const u32 num_verts,
               const MaterialId &material_id, const std::optional<Emitter> &emitter,
               FloatTexture *const alpha)
        : indices(indices), num_indices(num_indices), pos(pos), normals(normals),
          uvs(uvs), num_verts(num_verts), material_id(material_id), emitter(emitter),
          alpha(alpha) {}

    u32 *indices;
    u32 num_indices;
    point3 *pos;
    vec3 *normals{nullptr}; // may be null
    vec2 *uvs{nullptr};     // may be null
    u32 num_verts;
    MaterialId material_id;
    std::optional<Emitter> emitter;
    FloatTexture *alpha{nullptr}; // may be null
};

// SOA layout
struct Meshes {
    ShapeLightSample
    sample(ShapeIndex si, const vec3 &sample) const;

    std::vector<Mesh> meshes;
};

// Used only for sphere creation
struct SphereParams {
    SphereParams(const point3 &center, const f32 radius, const MaterialId &material_id,
                 const std::optional<Emitter> &emitter, FloatTexture *const alpha)
        : center(center), radius(radius), material_id(material_id), emitter(emitter),
          alpha(alpha) {}

    point3 center;
    f32 radius;
    MaterialId material_id;
    std::optional<Emitter> emitter;
    FloatTexture *alpha{nullptr}; // may be null
};

struct SphereVertex {
    SphereVertex(const point3 &pos, const f32 radius) : pos(pos), radius(radius) {}

    point3 pos;
    f32 radius;
};

struct SphereAttribs {
    bool has_light;
    u32 light_id;
    MaterialId material_id;
    FloatTexture *alpha;
};

// semi-SOA layout...
struct Spheres {
    void
    add_sphere(const SphereParams &sp, std::optional<u32> light_id);

    u32
    num_spheres() const {
        return vertices.size();
    }

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

    std::vector<SphereVertex> vertices;
    std::vector<SphereAttribs> attribs;
};

struct InstancedObj {
    Meshes meshes{};
    Spheres spheres{};
};

/// InstancedObj is one intanced object (let's say a bush made of many leaves).
/// Then the instances themselves are stored in a SOA layout, so that it can be shared
/// with Embree.
/// 'indices' maps from the instances themselves to the instanced objects.
struct Instances {
    std::vector<InstancedObj> instanced_objs;
    std::vector<SquareMatrix4> world_from_instances;
    std::vector<SquareMatrix4> wfi_inv_trans;
    std::vector<u32> indices;
};

/// GeometryContainer contains all of the geometry in the scene (triangle meshes,
/// spheres), etc...
class GeometryContainer {
public:
    template <GeometryPod T>
    GeometryBlock<T>
    allocate_geom_data(const std::size_t count) {
        return m_geom_storage.allocate<T>(count);
    }

    template <GeometryPod T>
    void
    add_geom_data(GeometryBlock<T> &block) {
        m_geom_storage.add_geom_data(block);
    }

    void
    add_mesh(const MeshParams &mp, std::optional<u32> lights_start_id,
             std::optional<InstanceId> inst_id);

    void
    add_sphere(const SphereParams &sp, std::optional<u32> light_id,
               std::optional<InstanceId> inst_id);

    InstanceId
    init_instance();

    void
    add_instanced_instance(InstanceId instance, const SquareMatrix4 &world_from_instance);

    /// Based on the shape type, returns the  index of the *next* shape in that category.
    u32
    get_next_shape_index(ShapeType type) const;

    ShapeLightSample
    sample_shape(ShapeIndex si, const point3 &pos, const vec3 &sample) const;

    f32
    shape_area(ShapeIndex si) const;

    const Meshes &
    meshes() const {
        return m_meshes;
    }

    const Spheres &
    spheres() const {
        return m_spheres;
    }

    const Instances &
    instances() const {
        return m_instances;
    }

private:
    Meshes m_meshes{};
    Spheres m_spheres{};

    Instances m_instances{};

    GeometryStorage m_geom_storage;
};

#endif // PT_GEOMETRY_CONTAINER_H
