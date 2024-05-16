
#include "geometry.h"

#include "../math/sampling.h"

void
Geometry::add_mesh(const MeshParams &mp, const Option<u32> lights_start_id) {
    auto mesh = Mesh(mp, lights_start_id);
    meshes.meshes.push_back(std::move(mesh));
}

void
Geometry::add_sphere(SphereParams sp, const Option<u32> light_id) {
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
Geometry::get_next_shape_index(const ShapeType type) const {
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
Geometry::sample_shape(const ShapeIndex si, const point3 &pos, const vec3 &sample) const {
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
Geometry::shape_area(const ShapeIndex si) const {
    switch (si.type) {
    case ShapeType::Mesh: {
        const auto &mesh = meshes.meshes[si.index];
        return mesh.tri_area(si.triangle_index);
    }
    case ShapeType::Sphere:
        return spheres.calc_sphere_area(si.index);
    default:
        assert(false);
    }
}

Mesh::
Mesh(const MeshParams &mp, const Option<u32> p_lights_start_id)
    : num_verts{mp.num_verts}, num_indices{mp.num_indices}, pos{mp.pos},
      normals{mp.normals}, uvs{mp.uvs}, indices{mp.indices}, alpha{mp.alpha},
      material_id{mp.material_id} {
    if (p_lights_start_id.has_value()) {
        lights_start_id = p_lights_start_id.value();
        has_light = true;
    }
}

u32
Mesh::num_triangles() const {
    return num_indices / 3;
}

uvec3
Mesh::get_tri_indices(const u32 triangle) const {
    const auto index = triangle * 3;
    const auto i0 = indices[index];
    const auto i1 = indices[index + 1];
    const auto i2 = indices[index + 2];
    return {i0, i1, i2};
}

std::array<point3, 3>
Mesh::get_tri_pos(const uvec3 &tri_indices) const {
    const auto p0 = pos[tri_indices[0]];
    const auto p1 = pos[tri_indices[1]];
    const auto p2 = pos[tri_indices[2]];
    return {p0, p1, p2};
}

f32
Mesh::tri_area(const u32 triangle) const {
    const auto tri_indices = get_tri_indices(triangle);
    const auto [p0, p1, p2] = get_tri_pos(tri_indices);
    const auto v1 = p1 - p0;
    const auto v2 = p2 - p0;
    const auto cross = vec3::cross(v1, v2);
    return cross.length() / 2.f;
}

norm_vec3
Mesh::calc_normal(const u32 triangle, const vec3 bar,
                  const bool want_geometric_normal) const {
    const auto tri_indices = get_tri_indices(triangle);

    if (normals && !want_geometric_normal) {
        const auto n0 = normals[tri_indices[0]];
        const auto n1 = normals[tri_indices[1]];
        const auto n2 = normals[tri_indices[2]];
        return barycentric_interp(bar, n0, n1, n2).normalized();
    } else {
        const auto [p0, p1, p2] = get_tri_pos(tri_indices);

        const auto v0 = p1 - p0;
        const auto v1 = p2 - p0;
        norm_vec3 normal = vec3::cross(v0, v1).normalized();
        if (normal.any_nan()) {
            // TODO: Degenerate triangle hack...
            normal = vec3(0.0f, 0.0f, 1.0f).normalized();
        }

        return normal;
    }
}

vec2
Mesh::calc_uvs(const u32 triangle_index, const vec3 &bar) const {
    // Idk what's suppossed to happen here without explicit UVs..
    const auto tri_indices = get_tri_indices(triangle_index);
    auto uv = vec2(0.);
    if (uvs) {
        const auto uv0 = uvs[tri_indices[0]];
        const auto uv1 = uvs[tri_indices[1]];
        const auto uv2 = uvs[tri_indices[2]];
        uv = barycentric_interp(bar, uv0, uv1, uv2);
    }

    return uv;
}

ShapeSample
Meshes::sample(ShapeIndex si, const vec3 &sample) const {
    const auto &mesh = meshes[si.index];

    const auto bar = sample_uniform_triangle(vec2(sample.y, sample.z));
    const auto tri_indices = mesh.get_tri_indices(si.triangle_index);
    const auto tri_pos = mesh.get_tri_pos(tri_indices);
    const auto sampled_pos = barycentric_interp(bar, tri_pos[0], tri_pos[1], tri_pos[2]);

    const auto normal = mesh.calc_normal(si.triangle_index, bar, false);
    const auto area = mesh.tri_area(si.triangle_index);

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
