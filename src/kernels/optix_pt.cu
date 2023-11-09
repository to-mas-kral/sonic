#include "optix_pt.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>

#include "../geometry/intersection.h"
#include "../integrator/utils.h"
#include "../utils/blue_noise_sampler.h"
#include "../utils/numtypes.h"
#include "raygen.h"

extern "C" {
__constant__ PtParams params;
}

const u32 NO_HIT = 0;
const u32 HIT_TRIANGLE = 1;
const u32 HIT_SPHERE = 2;

OpIntersection get_sphere_its(u32 index, u32 id, u32 id_1, u32 light, Spheres &spheres);
static __forceinline__ __device__ void set_payload_miss() { optixSetPayload_0(NO_HIT); }

static __forceinline__ __device__ void
set_payload_hit_triangle(u32 prim_index, u32 mesh_id, float2 barycentrics,
                         CUdeviceptr pos, CUdeviceptr indices, CUdeviceptr normals,
                         CUdeviceptr uvs) {
    // TODO: could pack hit/miss into the prim_index...
    optixSetPayload_0(HIT_TRIANGLE);
    optixSetPayload_1(prim_index);
    optixSetPayload_2(mesh_id);

    optixSetPayload_3(__float_as_uint(barycentrics.x));
    optixSetPayload_4(__float_as_uint(barycentrics.y));

    optixSetPayload_5(static_cast<u32>(pos));
    optixSetPayload_6((static_cast<u64>(pos) & 0xFFFF'FFFF'0000'0000U) >> 32U);
    optixSetPayload_7(static_cast<u32>(indices));
    optixSetPayload_8((static_cast<u64>(indices) & 0xFFFF'FFFF'0000'0000U) >> 32U);
    optixSetPayload_9(static_cast<u32>(normals));
    optixSetPayload_10((static_cast<u64>(normals) & 0xFFFF'FFFF'0000'0000U) >> 32U);
    optixSetPayload_11(static_cast<u32>(uvs));
    optixSetPayload_12((static_cast<u64>(uvs) & 0xFFFF'FFFF'0000'0000U) >> 32U);
}

static __forceinline__ __device__ void set_payload_hit_sphere(u32 prim_index,
                                                              u32 material_id,
                                                              u32 light_id,
                                                              bool has_light, f32 t) {
    optixSetPayload_0(HIT_SPHERE);
    optixSetPayload_1(prim_index);
    optixSetPayload_2(material_id);
    optixSetPayload_3(light_id);
    optixSetPayload_4(has_light);
    optixSetPayload_5(__float_as_uint(t));
}

static __forceinline__ __device__ OpIntersection get_triangle_its(
    u32 bar_y, u32 bar_z, u32 prim_index, u32 mesh_id, Ray &ray, CUdeviceptr d_pos,
    CUdeviceptr d_indices, CUdeviceptr d_normals, CUdeviceptr d_uvs) {

    f32 bar_y_f = __uint_as_float(bar_y);
    f32 bar_z_f = __uint_as_float(bar_z);
    vec3 bar = vec3(1.f - bar_y_f - bar_z_f, bar_y_f, bar_z_f);

    auto mesh_o = &params.meshes[mesh_id];

    auto has_normals = mesh_o->has_normals;
    auto has_uvs = mesh_o->has_uvs;
    auto has_light = mesh_o->has_light;
    auto light_id = mesh_o->light_id;
    auto material_id = mesh_o->material_id;

    vec3 *positions = (vec3 *)d_pos;
    u32 *indices = (u32 *)d_indices;
    vec3 *normals = (vec3 *)d_normals;
    vec2 *uvs = (vec2 *)d_uvs;

    u32 i0 = indices[3 * prim_index];
    u32 i1 = indices[3 * prim_index + 1];
    u32 i2 = indices[3 * prim_index + 2];

    vec3 p0 = positions[i0];
    vec3 p1 = positions[i1];
    vec3 p2 = positions[i2];

    // TODO: function for barycentric interpolation
    vec3 pos = bar.x * p0 + bar.y * p1 + bar.z * p2;

    vec3 normal;
    if (has_normals) {
        vec3 n0 = normals[i0];
        vec3 n1 = normals[i1];
        vec3 n2 = normals[i2];
        normal = glm::normalize(bar.x * n0 + bar.y * n1 + bar.z * n2);
    } else {
        vec3 v0 = p1 - p0;
        vec3 v1 = p2 - p0;
        normal = glm::normalize(cross(v0, v1));
        if (glm::any(glm::isnan(normal))) {
            // Degenerate triangle...
            // TODO: HACK
            normal = glm::normalize(-ray.dir);
        }
    }

    vec2 uv = vec2(0.);
    if (has_uvs) {
        vec2 uv0 = uvs[i0];
        vec2 uv1 = uvs[i1];
        vec2 uv2 = uvs[i2];
        uv = bar.x * uv0 + bar.y * uv1 + bar.z * uv2;
    }

    return OpIntersection{
        .material_id = material_id,
        .light_id = light_id,
        .has_light = has_light,
        .normal = normal,
        .pos = pos,
        .uv = uv,
    };
}

__device__ __forceinline__ CUdeviceptr unpack_ptr(u32 hi, u32 lo) {
    return (static_cast<u64>(hi) << 32U) | static_cast<u64>(lo);
}

__device__ __forceinline__ OpIntersection get_sphere_its(u32 sphere_index,
                                                         u32 material_id, u32 light_id,
                                                         u32 has_light, Spheres &spheres,
                                                         vec3 pos) {
    vec3 center = spheres.centers[sphere_index];
    f32 radius = spheres.radiuses[sphere_index];

    vec3 normal = glm::normalize(pos - center);

    // TODO: Sphere UV mapping could be wrong, test...
    // (1 / 2pi, 1 / pi)
    const vec2 pi_reciprocals = vec2(0.1591f, 0.3183f);
    vec2 uv = vec2(atan2(-normal.z, -normal.x), asin(normal.y));
    uv *= pi_reciprocals;
    uv += 0.5;

    return OpIntersection{
        .material_id = material_id,
        .light_id = light_id,
        .has_light = bool(has_light),
        .normal = normal,
        .pos = pos,
        .uv = uv,
    };
}

extern "C" __global__ void __raygen__rg() {
    const uint3 pixel = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    auto rc = params.rc;
    auto pixel_index = ((dim.y - 1U - pixel.y) * dim.x) + pixel.x;

    // auto sampler = BlueNoiseSampler();

    auto rand_state = &params.fb->get_rand_state()[pixel_index];

    /*auto [cam_s, cam_t] =
        sampler.create_samples<2>(pixel.x, pixel.y, (i32)params.sample_index);*/

    auto cam_sample = vec2(rng(rand_state), rng(rand_state));

    auto ray = gen_ray(pixel.x, pixel.y, dim.x, dim.y, cam_sample, rc);

    u32 depth = 1;
    vec3 throughput = vec3(1.f);
    vec3 radiance = vec3(0.f);

    while (true) {
        float3 raydir = make_float3(ray.dir.x, ray.dir.y, ray.dir.z);
        float3 rayorig = make_float3(ray.o.x, ray.o.y, ray.o.z);

        u32 p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12;
        optixTrace(params.gas_handle, rayorig, raydir, 0.0f, 1e16f, 0.0f,
                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, p0,
                   p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12);

        u32 did_hit = p0;

        if (did_hit) {
            /* auto [brdf_u, brdf_v, rr_sample] =
                sampler.create_samples<3>(pixel.x, pixel.y, (i32)params.sample_index); */
            auto bsdf_sample = vec2(rng(rand_state), rng(rand_state));
            f32 rr_sample = rng(rand_state);

            OpIntersection its;
            if (did_hit == HIT_TRIANGLE) {
                u32 prim_index = p1;
                u32 mesh_id = p2;
                u32 bar_y = p3;
                u32 bar_z = p4;
                u32 pos_lo = p5;
                u32 pos_hi = p6;
                u32 indices_lo = p7;
                u32 indices_hi = p8;
                u32 normals_lo = p9;
                u32 normals_hi = p10;
                u32 uvs_lo = p11;
                u32 uvs_hi = p12;

                CUdeviceptr d_pos = unpack_ptr(pos_hi, pos_lo);
                CUdeviceptr d_indices = unpack_ptr(indices_hi, indices_lo);
                CUdeviceptr d_normals = unpack_ptr(normals_hi, normals_lo);
                CUdeviceptr d_uvs = unpack_ptr(uvs_hi, uvs_lo);

                its = get_triangle_its(bar_y, bar_z, prim_index, mesh_id, ray, d_pos,
                                       d_indices, d_normals, d_uvs);
            } else {
                // Sphere
                u32 sphere_index = p1;
                u32 material_id = p2;
                u32 light_id = p3;
                u32 has_light = p4;
                f32 t = __uint_as_float(p5);

                Spheres &spheres = rc->geometry.spheres;
                vec3 pos = ray.o + ray.dir * t;

                its = get_sphere_its(sphere_index, material_id, light_id, has_light,
                                     spheres, pos);
            }

            auto material = &params.materials[its.material_id];

            vec3 emission = vec3(0.f);
            if (its.has_light) {
                emission = params.lights[its.light_id].emission();
            }

            if (glm::dot(-ray.dir, its.normal) < 0.f) {
                its.normal = -its.normal;
                emission = vec3(0.f);
            }

            // TODO: refactor this...
            Intersection old_its{
                .pos = its.pos,
                .normal = its.normal,
                .t = -1.f,       // TODO: t used for anything ?
                .mesh = nullptr, // TODO: own Intersection struct for OptiX pt...
            };

            vec3 sample_dir = material->sample(its.normal, -ray.dir, bsdf_sample);
            // TODO: what to do when cos_theta is 0 ? this minimum value is a band-aid
            // solution...
            f32 cos_theta = max(glm::dot(its.normal, sample_dir), 0.0001f);

            f32 pdf = material->pdf(cos_theta);
            vec3 brdf = material->eval(material, params.textures, its.uv);

            radiance += throughput * emission;
            throughput *= brdf * cos_theta * (1.f / pdf);

            auto rr = russian_roulette(depth, rr_sample, throughput);
            if (!rr.has_value()) {
                break;
            }

            auto roulette_compensation = rr.value();
            throughput *= 1.f / roulette_compensation;

            Ray new_ray = spawn_ray(old_its, sample_dir);
            ray = new_ray;
            depth++;
        } else {
            // TODO: move into miss program to reduce divergence ?
            // Ray has escaped the scene
            if (!rc->has_envmap) {
                break;
            } else {
                const Envmap *envmap = &rc->envmap;
                vec3 envrad = envmap->sample(ray);
                radiance += throughput * envrad;
                break;
            }
        }
    }

    params.fb->get_pixels()[pixel_index] += radiance;
}

extern "C" __global__ void __miss__ms() { set_payload_miss(); }

/*
 * From Nvidia docs:
 * "It is generally more efficient to have one hit shader handle multiple primitive
 * types (by switching on the value of optixGetPrimitiveType), rather than have several
 * hit shaders that implement the same ray behavior but differ only in the type of
 * geometry they expect."
 * */
extern "C" __global__ void __closesthit__ch() {
    PtHitGroupData *hit_data =
        reinterpret_cast<PtHitGroupData *>(optixGetSbtDataPointer());

    auto type = optixGetPrimitiveType();

    if (type == OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
        // Triangles
        const float2 barycentrics = optixGetTriangleBarycentrics();
        const u32 prim_index = optixGetPrimitiveIndex();

        set_payload_hit_triangle(prim_index, hit_data->mesh.mesh_id, barycentrics,
                                 hit_data->mesh.pos, hit_data->mesh.indices,
                                 hit_data->mesh.normals, hit_data->mesh.uvs);
    } else {
        // Spheres
        // TODO: maybe add sphere_id to hit_data... would be safer if multiple
        // sphere GASes are used in the future...
        const u32 sphere_index = optixGetSbtGASIndex();
        f32 t = optixGetRayTmax();

        set_payload_hit_sphere(sphere_index, hit_data->sphere.material_id,
                               hit_data->sphere.light_id, hit_data->sphere.has_light, t);
    }
}
