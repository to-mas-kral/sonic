#ifndef PT_GEOMETRY_H
#define PT_GEOMETRY_H

#include "../math/sampling.h"
#include "../math/vecmath.h"
#include "../scene/emitter.h"
#include "../utils/basic_types.h"

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
    get_tri_indices(u32 mesh_indices_index, u32 triangle) const {
        u32 index = triangle * 3;
        u32 i0 = indices[mesh_indices_index + index];
        u32 i1 = indices[mesh_indices_index + index + 1];
        u32 i2 = indices[mesh_indices_index + index + 2];
        return {i0, i1, i2};
    };

    Array<point3, 3>
    get_tri_pos(u32 mesh_pos_index, const Array<u32, 3> &tri_indices) const {
        point3 p0 = pos[mesh_pos_index + tri_indices[0]];
        point3 p1 = pos[mesh_pos_index + tri_indices[1]];
        point3 p2 = pos[mesh_pos_index + tri_indices[2]];
        return {p0, p1, p2};
    };

    f32
    calc_tri_area(u32 mesh_indices_index, u32 mesh_pos_index, u32 triangle) const {
        auto tri_indices = get_tri_indices(mesh_indices_index, triangle);
        const auto [p0, p1, p2] = get_tri_pos(mesh_pos_index, tri_indices);
        vec3 v1 = p1 - p0;
        vec3 v2 = p2 - p0;
        vec3 cross = vec3::cross(v1, v2);
        return cross.length() / 2.f;
    };

    norm_vec3
    calc_normal(bool has_normals, u32 i0, u32 i1, u32 i2, u32 normals_index,
                const vec3 &bar, const point3 &p0, const point3 &p1, const point3 &p2,
                bool want_geometric_normal = false) const {
        if (has_normals && !want_geometric_normal) {
            vec3 n0 = normals[normals_index + i0];
            vec3 n1 = normals[normals_index + i1];
            vec3 n2 = normals[normals_index + i2];
            return barycentric_interp(bar, n0, n1, n2).normalized();
        } else {
            vec3 v0 = p1 - p0;
            vec3 v1 = p2 - p0;
            norm_vec3 normal = vec3::cross(v0, v1).normalized();
            if (normal.any_nan()) {
                // TODO: Degenerate triangle hack...
                normal = vec3(0.5f, 0.3f, -0.7f).normalized();
            }

            return normal;
        }
    }

    vec2
    calc_uvs(bool has_uvs, u32 i0, u32 i1, u32 i2, u32 uvs_index, const vec3 &bar) const {
        // Idk what's suppossed to happen here without explicit UVs..
        vec2 uv = vec2(0.);
        if (has_uvs) {
            vec2 uv0 = uvs[uvs_index + i0];
            vec2 uv1 = uvs[uvs_index + i1];
            vec2 uv2 = uvs[uvs_index + i2];
            uv = barycentric_interp(bar, uv0, uv1, uv2);
        }

        return uv;
    }

    ShapeSample
    sample(ShapeIndex si, const vec3 &sample) const {
        auto &mesh = meshes[si.index];

        const vec3 bar = sample_uniform_triangle(vec2(sample.y, sample.z));
        auto tri_indices = get_tri_indices(mesh.indices_index, si.triangle_index);
        const auto tri_pos = get_tri_pos(mesh.pos_index, tri_indices);
        point3 sampled_pos = barycentric_interp(bar, tri_pos[0], tri_pos[1], tri_pos[2]);

        norm_vec3 normal =
            calc_normal(mesh.has_normals, tri_indices[0], tri_indices[1], tri_indices[2],
                        mesh.normals_index, bar, tri_pos[0], tri_pos[1], tri_pos[2]);

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
    sample(u32 index, const point3 &illuminated_pos, const vec3 &sample) const {
        vec3 sample_dir = sample_uniform_sphere(vec2(sample.x, sample.y));
        point3 center = vertices[index].pos;
        f32 radius = vertices[index].radius;

        point3 pos = center + radius * sample_dir;
        f32 area = calc_sphere_area(radius);

        return ShapeSample{
            .pos = pos,
            .normal = calc_normal(pos, center),
            .pdf = 1.f / area,
            .area = area,
        };
    }

    static f32
    calc_sphere_area(f32 radius) {
        return 4.f * M_PIf * sqr(radius);
    }

    f32
    calc_sphere_area(u32 sphere_id) const {
        f32 radius = vertices[sphere_id].radius;
        return calc_sphere_area(radius);
    }

    static norm_vec3
    calc_normal(const point3 &pos, const point3 &center,
                bool want_geometric_normal = false) {
        // TODO: geometric normals calculation when using normal mapping
        return (pos - center).normalized();
    }

    static vec2
    calc_uvs(const vec3 &normal) {
        // TODO: Sphere UV mapping could be wrong, test...
        // (1 / 2pi, 1 / pi)
        const vec2 pi_reciprocals = vec2(0.1591f, 0.3183f);
        vec2 uv = vec2(std::atan2(-normal.z, -normal.x), std::asin(normal.y));
        uv *= pi_reciprocals;
        uv += 0.5;
        return uv;
    }
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
    get_next_shape_index(ShapeType type) const {
        switch (type) {
        case ShapeType::Mesh:
            return meshes.meshes.size();
        case ShapeType::Sphere:
            return spheres.num_spheres;
        default:
            assert(false);
        }
    }

    ShapeSample
    sample_shape(ShapeIndex si, const point3 &pos, const vec3 &sample) const {
        switch (si.type) {
        case ShapeType::Mesh:
            return meshes.sample(si, sample);
        case ShapeType::Sphere:
            return spheres.sample(si.index, pos, sample);
        default:
            assert(false);
        }
    }

    f32
    shape_area(ShapeIndex si) const {
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
