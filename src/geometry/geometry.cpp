
#include "geometry.h"

#include "../math/sampling.h"

void
Geometry::add_mesh(const MeshParams &mp, Option<u32> lights_start_id) {
    u32 num_indices = mp.indices->size();
    u32 num_vertices = mp.pos->size();

    // TODO: accidentally quadratic resizing...

    u32 indices_index = meshes.indices.size();
    for (u32 &index : *mp.indices) {
        meshes.indices.push_back(index);
    }

    u32 pos_index = meshes.pos.size();
    for (auto &pos : *mp.pos) {
        meshes.pos.push_back(pos);
    }

    Option<u32> normals_index = {};
    if (mp.normals != nullptr) {
        normals_index = {meshes.normals.size()};
        assert(mp.normals->size() == mp.pos->size());

        for (auto normal : *mp.normals) {
            meshes.normals.push_back(normal);
        }
    }

    Option<u32> uvs_index = {};
    if (mp.uvs != nullptr) {
        uvs_index = {meshes.uvs.size()};
        assert(mp.uvs->size() == mp.pos->size());

        for (auto uv : *mp.uvs) {
            meshes.uvs.push_back(uv);
        }
    }

    auto mesh = Mesh(indices_index, pos_index, mp.material_id, lights_start_id,
                     num_indices, num_vertices, normals_index, uvs_index);
    meshes.meshes.push_back(mesh);
}

void
Geometry::add_sphere(SphereParams sp, Option<u32> light_id) {
    spheres.vertices.push_back(SphereVertex{
        .pos = sp.center,
        .radius = sp.radius,
    });
    spheres.material_ids.push_back(sp.material_id);
    spheres.has_light.push_back(light_id.has_value());
    if (light_id.has_value()) {
        spheres.light_ids.push_back(light_id.value());
    } else {
        spheres.light_ids.push_back(0);
    }

    spheres.num_spheres++;
}

u32
Geometry::get_next_shape_index(ShapeType type) const {
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
Geometry::sample_shape(ShapeIndex si, const point3 &pos, const vec3 &sample) const {
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
Geometry::shape_area(ShapeIndex si) const {
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

Mesh::Mesh(u32 indices_index, u32 pos_index, u32 material_id,
           Option<u32> p_lights_start_id, u32 num_indices, u32 num_vertices,
           Option<u32> p_normals_index, Option<u32> p_uvs_index)
    : indices_index(indices_index), pos_index(pos_index), material_id(material_id),
      num_indices(num_indices), num_vertices(num_vertices) {

    if (p_lights_start_id.has_value()) {
        lights_start_id = p_lights_start_id.value();
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

Array<u32, 3>
Meshes::get_tri_indices(u32 mesh_indices_index, u32 triangle) const {
    u32 index = triangle * 3;
    u32 i0 = indices[mesh_indices_index + index];
    u32 i1 = indices[mesh_indices_index + index + 1];
    u32 i2 = indices[mesh_indices_index + index + 2];
    return {i0, i1, i2};
}

Array<point3, 3>
Meshes::get_tri_pos(u32 mesh_pos_index, const Array<u32, 3> &tri_indices) const {
    point3 p0 = pos[mesh_pos_index + tri_indices[0]];
    point3 p1 = pos[mesh_pos_index + tri_indices[1]];
    point3 p2 = pos[mesh_pos_index + tri_indices[2]];
    return {p0, p1, p2};
}

f32
Meshes::calc_tri_area(u32 mesh_indices_index, u32 mesh_pos_index, u32 triangle) const {
    auto tri_indices = get_tri_indices(mesh_indices_index, triangle);
    const auto [p0, p1, p2] = get_tri_pos(mesh_pos_index, tri_indices);
    vec3 v1 = p1 - p0;
    vec3 v2 = p2 - p0;
    vec3 cross = vec3::cross(v1, v2);
    return cross.length() / 2.f;
}

norm_vec3
Meshes::calc_normal(bool has_normals, u32 i0, u32 i1, u32 i2, u32 normals_index,
                    const vec3 &bar, const point3 &p0, const point3 &p1, const point3 &p2,
                    bool want_geometric_normal) const {
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
            normal = vec3(0.0f, 0.0f, 1.0f).normalized();
        }

        return normal;
    }
}

vec2
Meshes::calc_uvs(bool has_uvs, u32 i0, u32 i1, u32 i2, u32 uvs_index,
                 const vec3 &bar) const {
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
Meshes::sample(ShapeIndex si, const vec3 &sample) const {
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

ShapeSample
Spheres::sample(u32 index, const point3 &illuminated_pos, const vec3 &sample) const {
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

f32
Spheres::calc_sphere_area(f32 radius) {
    return 4.f * M_PIf * sqr(radius);
}

f32
Spheres::calc_sphere_area(u32 sphere_id) const {
    f32 radius = vertices[sphere_id].radius;
    return calc_sphere_area(radius);
}

norm_vec3
Spheres::calc_normal(const point3 &pos, const point3 &center,
                     bool want_geometric_normal) {
    // TODO: geometric normals calculation when using normal mapping
    return (pos - center).normalized();
}

vec2
Spheres::calc_uvs(const vec3 &normal) {
    // TODO: Sphere UV mapping could be wrong, test...
    // (1 / 2pi, 1 / pi)
    const vec2 pi_reciprocals = vec2(0.1591f, 0.3183f);
    vec2 uv = vec2(std::atan2(-normal.z, -normal.x), std::asin(normal.y));
    uv *= pi_reciprocals;
    uv += 0.5;
    return uv;
}
