
#include "geometry.h"

__host__ void Geometry::add_mesh(MeshParams mp) {
    u32 num_indices = mp.indices->len();
    u32 num_vertices = mp.pos->len();

    u32 indices_index = meshes.indices.len();
    for (int i = 0; i < mp.indices->len(); i++) {
        meshes.indices.push(std::move((*mp.indices)[i]));
    }

    u32 pos_index = meshes.pos.len();
    for (int i = 0; i < mp.pos->len(); i++) {
        meshes.pos.push(std::move((*mp.pos)[i]));
    }

    cuda::std::optional<u32> normals_index = cuda::std::nullopt;
    if (mp.normals != nullptr) {
        normals_index = {meshes.normals.len()};
        assert(mp.normals->len() == mp.pos->len());

        for (int i = 0; i < mp.normals->len(); i++) {
            meshes.normals.push(std::move((*mp.normals)[i]));
        }
    }

    cuda::std::optional<u32> uvs_index = cuda::std::nullopt;
    if (mp.uvs != nullptr) {
        uvs_index = {meshes.uvs.len()};
        assert(mp.uvs->len() == mp.pos->len());

        for (int i = 0; i < mp.uvs->len(); i++) {
            meshes.uvs.push(std::move((*mp.uvs)[i]));
        }
    }

    auto mesh_id = meshes.meshes.len();
    auto mesh = Mesh(indices_index, pos_index, mp.material_id, mp.light_id, &meshes,
                     num_indices, num_vertices, normals_index, uvs_index);
    meshes.meshes.push(std::move(mesh));

    for (int i = 0; i < mp.indices->len(); i += 3) {
        Triangle triangle = Triangle(i / 3, mesh_id);
        meshes.triangles.push(std::move(triangle));
    }
}

__host__ void Geometry::add_sphere(SphereParams sp) {
    spheres.centers.push(std::move(sp.center));
    spheres.radiuses.push(std::move(sp.radius));
    spheres.material_ids.push(std::move(sp.material_id));
    spheres.has_light.push(std::move(sp.light_id.has_value()));
    if (sp.light_id.has_value()) {
        spheres.light_ids.push(std::move(sp.light_id.value()));
    }

    spheres.num_spheres++;
}

__host__ void Geometry::fixup_geometry_pointers() {
    // Fixup the triangle-mesh pointers
    for (int i = 0; i < meshes.triangles.len(); i++) {
        u32 mesh_id = meshes.triangles[i].mesh_id;
        meshes.triangles[i].set_mesh(&meshes.meshes[mesh_id]);
    }
}

__host__ __device__ const u32 *Mesh::get_indices() const {
    return &tm->indices[indices_index];
}

__host__ __device__ const vec3 *Mesh::get_pos() const { return &tm->pos[pos_index]; }

__host__ __device__ cuda::std::array<u32, 3> Triangle::get_indices() {
    u32 i0 = mesh->get_indices()[id * 3];
    u32 i1 = mesh->get_indices()[id * 3 + 1];
    u32 i2 = mesh->get_indices()[id * 3 + 2];
    return cuda::std::array<u32, 3>{i0, i1, i2};
}

__host__ __device__ cuda::std::array<vec3, 3> Triangle::get_pos() {
    auto [i0, i1, i2] = get_indices();
    vec3 p0 = mesh->get_pos()[i0];
    vec3 p1 = mesh->get_pos()[i1];
    vec3 p2 = mesh->get_pos()[i2];
    return cuda::std::array<vec3, 3>{p0, p1, p2};
}

/// Möller-Trumbore intersection algorithm
__device__ cuda::std::optional<Intersection> Triangle::intersect(Ray &ray) {
    f32 eps = 0.0000001f;

    auto [p0, p1, p2] = get_pos();

    vec3 e1 = p1 - p0;
    vec3 e2 = p2 - p0;

    vec3 h = cross(ray.dir, e2);
    f32 a = dot(e1, h);

    if (a > -eps && a < eps) {
        return cuda::std::nullopt;
    }

    f32 f = 1.f / a;
    vec3 s = ray.o - p0;
    f32 u = f * dot(s, h);
    if (u < 0.f || u > 1.f) {
        return cuda::std::nullopt;
    }

    vec3 q = cross(s, e1);
    f32 v = f * dot(ray.dir, q);
    if (v < 0.f || u + v > 1.f) {
        return cuda::std::nullopt;
    }

    f32 t = f * dot(e2, q);
    if (t > eps) {
        vec3 pos = ray(t);

        // barycentric coords
        // f32 r = 1.f - u - v;

        // auto bar = {r, u, v};
        // auto [i0, i1, i2] = get_indices();

        // auto uv = mesh.uvs.as_ref().map(
        //     | uvs | barycentric_interp(&bar, &uvs[i0], &uvs[i1], &uvs[i2]));

        // TODO: adjust when mesh normals are added
        vec3 v0 = p1 - p0;
        vec3 v1 = p2 - p0;
        vec3 normal = glm::normalize(cross(v0, v1));

        Intersection its;
        its.pos = pos;
        its.normal = normal;
        its.t = t;
        its.mesh = mesh;
        return {its};
    }

    return cuda::std::nullopt;
}

__host__ AABB Triangle::aabb() {
    auto [p0, p1, p2] = get_pos();
    return AABB(p0, p1).union_point(p2);
}

void Triangle::set_mesh(Mesh *a_mesh) { mesh = a_mesh; }
