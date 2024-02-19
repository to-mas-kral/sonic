#ifndef PT_GEOMETRY_H
#define PT_GEOMETRY_H

#include "../math/sampling.h"
#include "../math/vecmath.h"
#include "../scene/emitter.h"
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

struct ShapeSample {
    point3 pos;
    norm_vec3 normal;
    f32 pdf;
    f32 area;
};

struct Mesh {
    Mesh(u32 indices_index, u32 pos_index, u32 material_id, Option<u32> p_lights_start_id,
         u32 num_indices, u32 num_vertices, Option<u32> p_normals_index,
         Option<u32> p_uvs_index);

    u32
    num_triangles() const {
        return num_indices / 3;
    };

    u32 pos_index;
    u32 indices_index;
    u32 normals_index;
    u32 uvs_index;

    u32 num_vertices;
    u32 num_indices;

    // Don't use optional<T> due too much memory consumption
    bool has_normals = false;
    bool has_uvs = false;
    bool has_light = false;
    u32 lights_start_id;
    u32 material_id;
};

// Used only for mesh creation
struct MeshParams {
    std::vector<u32> *indices;
    std::vector<point3> *pos;
    std::vector<vec3> *normals = nullptr; // may be null
    std::vector<vec2> *uvs = nullptr;     // may be null
    u32 material_id;
    Option<Emitter> emitter = {};
};

// SOA layout
struct Meshes {
    std::vector<Mesh> meshes{};

    std::vector<u32> indices{};
    std::vector<point3> pos{};
    std::vector<vec3> normals{};
    std::vector<vec2> uvs{};

    Array<u32, 3>
    get_tri_indices(u32 mesh_indices_index, u32 triangle) const;
    ;

    Array<point3, 3>
    get_tri_pos(u32 mesh_pos_index, const Array<u32, 3> &tri_indices) const;
    ;

    f32
    calc_tri_area(u32 mesh_indices_index, u32 mesh_pos_index, u32 triangle) const;
    ;

    norm_vec3
    calc_normal(bool has_normals, u32 i0, u32 i1, u32 i2, u32 normals_index,
                const vec3 &bar, const point3 &p0, const point3 &p1, const point3 &p2,
                bool want_geometric_normal = false) const;

    vec2
    calc_uvs(bool has_uvs, u32 i0, u32 i1, u32 i2, u32 uvs_index, const vec3 &bar) const;

    ShapeSample
    sample(ShapeIndex si, const vec3 &sample) const;
};

// Used only for sphere creation
struct SphereParams {
    point3 center;
    f32 radius;
    u32 material_id;
    Option<Emitter> emitter = {};
};

struct SphereVertex {
    point3 pos;
    f32 radius;
};

// SOA layout
struct Spheres {
    std::vector<SphereVertex> vertices{};
    std::vector<u32> material_ids{};
    std::vector<bool> has_light{};
    std::vector<u32> light_ids{};
    u32 num_spheres = 0;

    ShapeSample
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

struct Geometry {
    Meshes meshes{};
    Spheres spheres{};

    void
    add_mesh(const MeshParams &mp, Option<u32> lights_start_id);
    void
    add_sphere(SphereParams sp, Option<u32> light_id);

    /// Based on the shape type, returns the  index of the *next* shape in that category.
    u32
    get_next_shape_index(ShapeType type) const;

    ShapeSample
    sample_shape(ShapeIndex si, const point3 &pos, const vec3 &sample) const;

    f32
    shape_area(ShapeIndex si) const;
};

#endif // PT_GEOMETRY_H
