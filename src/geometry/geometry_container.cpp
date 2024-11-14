
#include "geometry_container.h"

#include "../math/sampling.h"

void
GeometryContainer::add_mesh(const MeshParams &mp,
                            const std::optional<u32> lights_start_id,
                            const std::optional<InstanceId> inst_id) {
    auto mesh = Mesh(mp, lights_start_id);

    if (inst_id.has_value()) {
        m_instances.instanced_objs[inst_id.value().inner].meshes.meshes.push_back(
            std::move(mesh));
    } else {
        m_meshes.meshes.push_back(std::move(mesh));
    }
}

void
GeometryContainer::add_sphere(const SphereParams &sp, const std::optional<u32> light_id,
                              const std::optional<InstanceId> inst_id) {

    if (inst_id.has_value()) {
        m_instances.instanced_objs[inst_id.value().inner].spheres.add_sphere(sp,
                                                                             light_id);
    } else {
        m_spheres.add_sphere(sp, light_id);
    }
}

InstanceId
GeometryContainer::init_instance() {
    const u32 id = m_instances.instanced_objs.size();
    m_instances.instanced_objs.emplace_back();
    return InstanceId{id};
}

void
GeometryContainer::add_instanced_instance(const InstanceId instance,
                                          const SquareMatrix4 &world_from_instance) {
    m_instances.indices.push_back(instance.inner);
    m_instances.world_from_instances.push_back(world_from_instance);
    m_instances.wfi_inv_trans.push_back(world_from_instance.inverse().transpose());
}

u32
GeometryContainer::get_next_shape_index(const ShapeType type) const {
    switch (type) {
    case ShapeType::Mesh:
        return m_meshes.meshes.size();
    case ShapeType::Sphere:
        return m_spheres.num_spheres();
    default:
        panic();
    }
}

ShapeLightSample
GeometryContainer::sample_shape(const ShapeIndex si, const point3 &pos,
                                const vec3 &sample) const {
    switch (si.type) {
    case ShapeType::Mesh:
        return m_meshes.sample(si, sample);
    case ShapeType::Sphere:
        return m_spheres.sample(si.index, pos, sample);
    default:
        panic();
    }
}

f32
GeometryContainer::shape_area(const ShapeIndex si) const {
    switch (si.type) {
    case ShapeType::Mesh: {
        const auto &mesh = m_meshes.meshes[si.index];
        return mesh.tri_area(si.triangle_index);
    }
    case ShapeType::Sphere:
        return m_spheres.calc_sphere_area(si.index);
    default:
        panic();
    }
}

Mesh::
Mesh(const MeshParams &mp, const std::optional<u32> p_lights_start_id)
    : m_num_verts{mp.num_verts}, m_num_indices{mp.num_indices}, m_pos{mp.pos},
      m_normals{mp.normals}, m_uvs{mp.uvs}, m_indices{mp.indices}, m_alpha{mp.alpha},
      m_material_id{mp.material_id} {
    if (p_lights_start_id.has_value()) {
        m_lights_start_id = p_lights_start_id.value();
        m_has_light = true;
    }
}

u32
Mesh::num_triangles() const {
    return m_num_indices / 3;
}

uvec3
Mesh::get_tri_indices(const u32 triangle) const {
    const auto index = triangle * 3;
    const auto i0 = m_indices[index];
    const auto i1 = m_indices[index + 1];
    const auto i2 = m_indices[index + 2];
    return {i0, i1, i2};
}

std::array<point3, 3>
Mesh::get_tri_pos(const uvec3 &tri_indices) const {
    const auto p0 = m_pos[tri_indices[0]];
    const auto p1 = m_pos[tri_indices[1]];
    const auto p2 = m_pos[tri_indices[2]];
    return {p0, p1, p2};
}

f32
Mesh::tri_area(const u32 triangle) const {
    const auto tri_indices = get_tri_indices(triangle);
    const auto [p0, p1, p2] = get_tri_pos(tri_indices);
    const auto v1 = p1 - p0;
    const auto v2 = p2 - p0;
    const auto cross = vec3::cross(v1, v2);
    return cross.length() / 2.F;
}

norm_vec3
Mesh::calc_normal(const u32 triangle, const vec3 bar,
                  const bool want_geometric_normal) const {
    const auto tri_indices = get_tri_indices(triangle);

    if (m_normals != nullptr && !want_geometric_normal) {
        const auto n0 = m_normals[tri_indices[0]];
        const auto n1 = m_normals[tri_indices[1]];
        const auto n2 = m_normals[tri_indices[2]];
        return barycentric_interp(bar, n0, n1, n2).normalized();
    } else {
        const auto [p0, p1, p2] = get_tri_pos(tri_indices);

        const auto v0 = p1 - p0;
        const auto v1 = p2 - p0;
        norm_vec3 normal = vec3::cross(v0, v1).normalized();
        if (normal.any_nan()) {
            // TODO: Degenerate triangle hack...
            normal = vec3(0.0F, 0.0F, 1.0F).normalized();
        }

        return normal;
    }
}

vec2
Mesh::calc_uvs(const u32 triangle_index, const vec3 &bar) const {
    // Idk what's suppossed to happen here without explicit UVs..
    const auto tri_indices = get_tri_indices(triangle_index);
    auto uv = vec2(0.);
    if (m_uvs != nullptr) {
        const auto uv0 = m_uvs[tri_indices[0]];
        const auto uv1 = m_uvs[tri_indices[1]];
        const auto uv2 = m_uvs[tri_indices[2]];
        uv = barycentric_interp(bar, uv0, uv1, uv2);
    }

    return uv;
}

Mesh::
Mesh(Mesh &&other) noexcept
    : m_num_verts(other.m_num_verts), m_num_indices(other.m_num_indices),
      m_pos(other.m_pos), m_normals(other.m_normals), m_uvs(other.m_uvs),
      m_indices(other.m_indices), m_alpha{other.m_alpha}, m_has_light(other.m_has_light),
      m_lights_start_id(other.m_lights_start_id), m_material_id(other.m_material_id) {
    other.m_pos = nullptr;
    other.m_normals = nullptr;
    other.m_uvs = nullptr;
    other.m_indices = nullptr;
}

Mesh &
Mesh::operator=(Mesh &&other) noexcept {
    if (this == &other) {
        return *this;
    }
    m_num_verts = other.m_num_verts;
    m_num_indices = other.m_num_indices;
    m_pos = other.m_pos;
    m_normals = other.m_normals;
    m_alpha = other.m_alpha;
    m_uvs = other.m_uvs;
    m_indices = other.m_indices;
    m_has_light = other.m_has_light;
    m_lights_start_id = other.m_lights_start_id;
    m_material_id = other.m_material_id;

    other.m_pos = nullptr;
    other.m_normals = nullptr;
    other.m_uvs = nullptr;
    other.m_indices = nullptr;

    return *this;
}

ShapeLightSample
Meshes::sample(const ShapeIndex si, const vec3 &sample) const {
    const auto &mesh = meshes[si.index];

    const auto bar = sample_uniform_triangle(vec2(sample.y, sample.z));
    const auto tri_indices = mesh.get_tri_indices(si.triangle_index);
    const auto tri_pos = mesh.get_tri_pos(tri_indices);
    const auto sampled_pos = barycentric_interp(bar, tri_pos[0], tri_pos[1], tri_pos[2]);

    const auto normal = mesh.calc_normal(si.triangle_index, bar, false);
    const auto area = mesh.tri_area(si.triangle_index);

    return ShapeLightSample{
        .pos = sampled_pos,
        .normal = normal,
        .pdf = 1.F / area,
    };
}

void
Spheres::add_sphere(const SphereParams &sp, const std::optional<u32> light_id) {
    vertices.emplace_back(sp.center, sp.radius);

    attribs.emplace_back(SphereAttribs{.has_light = light_id.has_value(),
                                       .light_id = light_id.value_or(0),
                                       .material_id = sp.material_id,
                                       .alpha = sp.alpha});
}

ShapeLightSample
Spheres::sample(const u32 index, const point3 &illuminated_pos,
                const vec3 &sample) const {
    const vec3 sample_dir = sample_uniform_sphere(vec2(sample.x, sample.y));
    const point3 center = vertices[index].pos;
    const f32 radius = vertices[index].radius;

    const point3 pos = center + radius * sample_dir;
    const f32 area = calc_sphere_area(radius);

    return ShapeLightSample{
        .pos = pos,
        .normal = calc_normal(pos, center),
        .pdf = 1.F / area,
    };
}

f32
Spheres::calc_sphere_area(const f32 radius) {
    return 4.F * M_PIf * sqr(radius);
}

f32
Spheres::calc_sphere_area(const u32 sphere_id) const {
    const f32 radius = vertices[sphere_id].radius;
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
    const auto pi_reciprocals = vec2(0.1591F, 0.3183F);
    auto uv = vec2(std::atan2(-normal.z, -normal.x), std::asin(normal.y));
    uv *= pi_reciprocals;
    uv += 0.5;
    return uv;
}
