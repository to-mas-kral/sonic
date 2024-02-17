#ifndef PT_INTERSECTION_H
#define PT_INTERSECTION_H

struct Intersection {
    __device__ static Intersection
    make_empty() {
        return Intersection{.material_id = 0,
                            .light_id = 0,
                            .has_light = false,
                            .normal{0.f, 1.f, 0.f},
                            .geometric_normal{0.f, 1.f, 0.f},
                            .pos{0.f, 0.f, 0.f},
                            .uv{0.f, 0.f}};
    }

    u32 material_id;
    u32 light_id;
    bool has_light;
    /// "shading" normal affected by interpolation or normal maps
    norm_vec3 normal;
    /// "true" normal, always perpendicular to the geometry, used for self-intersection
    /// avoidance
    norm_vec3 geometric_normal;
    point3 pos;
    vec2 uv;
};

static __forceinline__ __device__ Intersection
get_triangle_its(Mesh *meshes, Geometry *geometry, u32 bar_y, u32 bar_z,
                 u32 triangle_index, u32 mesh_id) {
    f32 bar_y_f = __uint_as_float(bar_y);
    f32 bar_z_f = __uint_as_float(bar_z);
    vec3 bar = vec3(1.f - bar_y_f - bar_z_f, bar_y_f, bar_z_f);

    auto mesh_o = &meshes[mesh_id];

    point3 *positions = &geometry->meshes.pos[mesh_o->pos_index];
    u32 *indices = &geometry->meshes.indices[mesh_o->indices_index];
    const vec3 *normals = geometry->meshes.normals.get_ptr_to(mesh_o->normals_index);
    const vec2 *uvs = geometry->meshes.uvs.get_ptr_to(mesh_o->uvs_index);

    u32 i0 = indices[3 * triangle_index];
    u32 i1 = indices[3 * triangle_index + 1];
    u32 i2 = indices[3 * triangle_index + 2];

    point3 p0 = positions[i0];
    point3 p1 = positions[i1];
    point3 p2 = positions[i2];

    point3 pos = barycentric_interp(bar, p0, p1, p2);

    norm_vec3 normal =
        Meshes::calc_normal(mesh_o->has_normals, i0, i1, i2, normals, bar, p0, p1, p2);
    norm_vec3 geometric_normal = Meshes::calc_normal(mesh_o->has_normals, i0, i1, i2,
                                                     normals, bar, p0, p1, p2, true);
    vec2 uv = Meshes::calc_uvs(mesh_o->has_uvs, i0, i1, i2, uvs, bar);

    return Intersection{
        .material_id = mesh_o->material_id,
        .light_id = mesh_o->lights_start_id + triangle_index,
        .has_light = mesh_o->has_light,
        .normal = normal,
        .geometric_normal = geometric_normal,
        .pos = pos,
        .uv = uv,
    };
}

__device__ __forceinline__ Intersection
get_sphere_its(u32 sphere_index, Spheres &spheres, const point3 &pos) {
    point3 center = spheres.centers[sphere_index];
    u32 material_id = spheres.material_ids[sphere_index];
    u32 light_id = spheres.light_ids[sphere_index];
    bool has_light = spheres.has_light[sphere_index];

    norm_vec3 normal = Spheres::calc_normal(pos, center);
    norm_vec3 geometric_normal = Spheres::calc_normal(pos, center, true);
    vec2 uv = Spheres::calc_uvs(normal);

    return Intersection{
        .material_id = material_id,
        .light_id = light_id,
        .has_light = bool(has_light),
        .normal = normal,
        .geometric_normal = geometric_normal,
        .pos = pos,
        .uv = uv,
    };
}

__device__ __forceinline__ Intersection
get_its(Mesh *meshes, Geometry *geometry, Scene *sc, u32 p1, u32 p2, u32 p3, u32 p4,
        HitType hit_type, const Ray &ray) {
    if (hit_type == HitType::Triangle) {
        u32 prim_index = p1;
        u32 mesh_id = p2;
        u32 bar_y = p3;
        u32 bar_z = p4;

        return get_triangle_its(meshes, geometry, bar_y, bar_z, prim_index, mesh_id);
    } else {
        // Sphere
        u32 sphere_index = p1;
        f32 t = __uint_as_float(p2);

        Spheres &spheres = sc->geometry.spheres;
        point3 pos = ray.at(t);

        return get_sphere_its(sphere_index, spheres, pos);
    }
}

__device__ __forceinline__ constexpr float
origin() {
    return 1.0f / 32.0f;
}

__device__ __forceinline__ constexpr float
float_scale() {
    return 1.0f / 65536.0f;
}

__device__ __forceinline__ constexpr float
int_scale() {
    return 256.0f;
}

// Taken from GPU Gems - Chapter 6 - A Fast and Robust Method for Avoiding
// Self-Intersection - Carsten Wächter and Nikolaus Binder - NVIDIA
/// Normal points outward for rays exiting the surface, else is flipped.
__device__ __forceinline__ point3
offset_ray(const point3 &p, const norm_vec3 &n) {
    ivec3 of_i(int_scale() * n.x, int_scale() * n.y, int_scale() * n.z);

    point3 p_i(__int_as_float(__float_as_int(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
               __int_as_float(__float_as_int(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
               __int_as_float(__float_as_int(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

    return point3(fabsf(p.x) < origin() ? p.x + float_scale() * n.x : p_i.x,
                  fabsf(p.y) < origin() ? p.y + float_scale() * n.y : p_i.y,
                  fabsf(p.z) < origin() ? p.z + float_scale() * n.z : p_i.z);
}

__device__ __forceinline__ Ray
spawn_ray(point3 &pos, const norm_vec3 &spawn_ray_normal, const norm_vec3 &wi) {
    point3 offset_orig = offset_ray(pos, spawn_ray_normal);
    return Ray(offset_orig, wi);
}

#endif // PT_INTERSECTION_H
