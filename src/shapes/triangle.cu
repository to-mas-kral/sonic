
#include "../render_context_common.h"

__device__ bool Triangle::intersect(Intersection &its, Ray &ray) {
    f32 eps = 0.0000001f;

    auto [p0, p1, p2] = get_pos();

    vec3 e1 = p1 - p0;
    vec3 e2 = p2 - p0;

    vec3 h = cross(ray.dir, e2);
    f32 a = dot(e1, h);

    if (a > -eps && a < eps) {
        return false;
    }

    f32 f = 1.f / a;
    vec3 s = ray.o - p0;
    f32 u = f * dot(s, h);
    if (u < 0.f || u > 1.f) {
        return false;
    }

    vec3 q = cross(s, e1);
    f32 v = f * dot(ray.dir, q);
    if (v < 0.f || u + v > 1.f) {
        return false;
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

        its.pos = pos;
        its.normal = normal;
        its.t = t;
        its.mesh = mesh;
        return true;
    }

    return false;
}

__host__ __device__ cuda::std::tuple<vec3, vec3, vec3> Triangle::get_pos() {
    auto [i0, i1, i2] = get_indices();
    vec3 p0 = mesh->get_pos()[i0];
    vec3 p1 = mesh->get_pos()[i1];
    vec3 p2 = mesh->get_pos()[i2];
    return {p0, p1, p2};
}

__host__ __device__ cuda::std::tuple<u32, u32, u32> Triangle::get_indices() {
    u32 i0 = mesh->get_indices()[id * 3];
    u32 i1 = mesh->get_indices()[id * 3 + 1];
    u32 i2 = mesh->get_indices()[id * 3 + 2];
    return {i0, i1, i2};
}

__host__ AABB Triangle::aabb() {
    auto [p0, p1, p2] = get_pos();
    return AABB(p0, p1).union_point(p2);
}

void Triangle::set_mesh(Mesh *a_mesh) { mesh = a_mesh; }
