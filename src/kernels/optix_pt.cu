#include "optix_pt.h"

#include <cuda_runtime.h>
#include <glm/gtx/norm.hpp>
#include <optix.h>
#include <optix_device.h>

#include "../integrator/utils.h"
#include "../utils/numtypes.h"
#include "../utils/sampler.h"
#include "raygen.h"

extern "C" {
__constant__ PtParams params;
}

const u32 NO_HIT = 0;
const u32 HIT_TRIANGLE = 1;
const u32 HIT_SPHERE = 2;

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

static __forceinline__ __device__ Intersection
get_triangle_its(u32 bar_y, u32 bar_z, u32 triangle_index, u32 mesh_id, CUdeviceptr d_pos,
                 CUdeviceptr d_indices, CUdeviceptr d_normals, CUdeviceptr d_uvs) {

    f32 bar_y_f = __uint_as_float(bar_y);
    f32 bar_z_f = __uint_as_float(bar_z);
    vec3 bar = vec3(1.f - bar_y_f - bar_z_f, bar_y_f, bar_z_f);

    auto mesh_o = &params.meshes[mesh_id];

    vec3 *positions = (vec3 *)d_pos;
    u32 *indices = (u32 *)d_indices;
    vec3 *normals = (vec3 *)d_normals;
    vec2 *uvs = (vec2 *)d_uvs;

    u32 i0 = indices[3 * triangle_index];
    u32 i1 = indices[3 * triangle_index + 1];
    u32 i2 = indices[3 * triangle_index + 2];

    vec3 p0 = positions[i0];
    vec3 p1 = positions[i1];
    vec3 p2 = positions[i2];

    vec3 pos = barycentric_interp(bar, p0, p1, p2);

    vec3 normal =
        Meshes::calc_normal(mesh_o->has_normals, i0, i1, i2, normals, bar, p0, p1, p2);
    vec2 uv = Meshes::calc_uvs(mesh_o->has_uvs, i0, i1, i2, uvs, bar);

    return Intersection{
        .material_id = mesh_o->material_id,
        .light_id = mesh_o->lights_start_id + triangle_index,
        .has_light = mesh_o->has_light,
        .normal = normal,
        .pos = pos,
        .uv = uv,
    };
}

__device__ __forceinline__ CUdeviceptr unpack_ptr(u32 hi, u32 lo) {
    return (static_cast<u64>(hi) << 32U) | static_cast<u64>(lo);
}

__device__ __forceinline__ Intersection get_sphere_its(u32 sphere_index,
                                                         u32 material_id, u32 light_id,
                                                         u32 has_light, Spheres &spheres,
                                                         const vec3 &pos) {
    vec3 center = spheres.centers[sphere_index];

    vec3 normal = Spheres::calc_normal(pos, center);
    vec2 uv = Spheres::calc_uvs(normal);

    return Intersection{
        .material_id = material_id,
        .light_id = light_id,
        .has_light = bool(has_light),
        .normal = normal,
        .pos = pos,
        .uv = uv,
    };
}

__device__ __forceinline__ Intersection get_its(Scene *sc, u32 p1, u32 p2, u32 p3,
                                                  u32 p4, u32 p5, u32 p6, u32 p7, u32 p8,
                                                  u32 p9, u32 p10, u32 p11, u32 p12,
                                                  u32 did_hit, Ray &ray) {
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

        return get_triangle_its(bar_y, bar_z, prim_index, mesh_id, d_pos, d_indices,
                                d_normals, d_uvs);
    } else {
        // Sphere
        u32 sphere_index = p1;
        u32 material_id = p2;
        u32 light_id = p3;
        u32 has_light = p4;
        f32 t = __uint_as_float(p5);

        Spheres &spheres = sc->geometry.spheres;
        vec3 pos = ray.o + ray.dir * t;

        return get_sphere_its(sphere_index, material_id, light_id, has_light, spheres,
                              pos);
    }
}

// Multisple Importance Sampling for lights
__device__ __forceinline__ void
light_mis(const Intersection &its, const Ray &traced_ray, const Ray &bxdf_ray,
          const LightSample &light_sample, const ShapeSample &shape_sample,
          const Material *material, vec3 *radiance, const vec3 &throughput) {
    vec3 light_pos = shape_sample.pos;
    vec3 pl_norm = glm::normalize(light_pos - its.pos);
    f32 pl_mag_sq = glm::length2(light_pos - its.pos);
    f32 cos_light = max(glm::dot(shape_sample.normal, -pl_norm), 0.000001f);

    auto sgeom_light = get_shading_geom(its.normal, pl_norm, -traced_ray.dir);

    // Quickly precheck if light is reachable
    if (sgeom_light.cos_theta > 0.f && cos_light > 0.f) {
        f32 pl_mag = glm::length(light_pos - its.pos);

        // Use the origin of the BXDF ray, which is already offset from the surface so
        // that it doesn't self-intersect.
        vec3 lrd = glm::normalize(light_pos - bxdf_ray.o);
        float3 raydir = make_float3(lrd.x, lrd.y, lrd.z);
        float3 rayorig = make_float3(bxdf_ray.o.x, bxdf_ray.o.y, bxdf_ray.o.z);
        u32 did_hit = 1;
        // TODO: didn!t get much of a speedup, investigate
        // https://www.willusher.io/graphics/2019/09/06/faster-shadow-rays-on-rtx
        optixTrace(params.gas_handle, rayorig, raydir, 0.0001f, pl_mag - 0.001f, 0.0f,
                   OptixVisibilityMask(255),
                   OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                       OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                   0, 1, 0, did_hit);

        if (!did_hit) {
            // Probability of sampling this light in terms of solid angle from the
            // probability distribution of the lights. Formula from
            // https://www.pbr-book.org/4ed/Radiometry,_Spectra,_and_Color/Working_with_Radiometric_Integrals#IntegralsoverArea
            f32 pdf_light = shape_sample.pdf * light_sample.pdf * (pl_mag_sq / cos_light);

            vec3 bxdf_light = material->eval(params.textures, its.uv);
            f32 weight_light = mis_power_heuristic(pdf_light, material->pdf(sgeom_light));

            vec3 light_emission = light_sample.light.emitter.emission();

            *radiance += bxdf_light * sgeom_light.cos_theta * (1.f / pdf_light) *
                         light_emission * weight_light * throughput;
        }
    }
}

extern "C" __global__ void __raygen__rg() {
    const uint3 pixel = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    auto rc = params.rc;
    auto sc = &rc->scene;
    auto pixel_index = ((dim.y - 1U - pixel.y) * dim.x) + pixel.x;

    auto sampler = &params.fb->get_rand_state()[pixel_index];

    auto cam_sample = vec2(sampler->sample(), sampler->sample());

    auto ray = gen_ray(pixel.x, pixel.y, dim.x, dim.y, cam_sample, rc);

    u32 depth = 1;
    vec3 throughput = vec3(1.f);
    vec3 radiance = vec3(0.f);

    vec3 last_hit_pos = vec3(0.f);
    f32 last_pdf_bxdf = 0.f;

    while (true) {
        float3 raydir = make_float3(ray.dir.x, ray.dir.y, ray.dir.z);
        float3 rayorig = make_float3(ray.o.x, ray.o.y, ray.o.z);

        u32 p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12;
        optixTrace(params.gas_handle, rayorig, raydir, 0.0f, 1e16f, 0.0f,
                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, p0,
                   p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12);

        u32 did_hit = p0;
        if (did_hit) {
            auto bsdf_sample = vec2(sampler->sample(), sampler->sample());
            auto rr_sample = sampler->sample();
            Intersection its = get_its(sc, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11,
                                         p12, did_hit, ray);

            auto material = &params.materials[its.material_id];
            bool is_frontfacing = glm::dot(-ray.dir, its.normal) > 0.f;

            if (its.has_light && is_frontfacing) {
                vec3 emission = params.lights[its.light_id].emitter.emission();

                if (depth == 1) {
                    // Primary ray hit, can apply MIS...
                    radiance += throughput * emission;
                } else {
                    vec3 pl_norm = glm::normalize(its.pos - last_hit_pos);
                    f32 pl_mag_sq = glm::length2(its.pos - last_hit_pos);
                    f32 cos_light = glm::dot(its.normal, -pl_norm);

                    // last_pdf_bxdf is the probability of this light having been sampled
                    // from the probability distribution of the BXDF of the *preceding*
                    // hit.

                    // TODO: currently calculating the shape PDF by assuming pdf = 1. /
                    // area will have to change with non-uniform sampling !
                    f32 light_area =
                        sc->geometry.shape_area(sc->lights[its.light_id].shape);

                    // pdf_light is the probability of this light being sampled from the
                    // probability distribution of the lights.
                    f32 pdf_light = sc->light_sampler.light_sample_pdf(its.light_id) *
                                    pl_mag_sq / (light_area * cos_light);

                    f32 bxdf_weight = mis_power_heuristic(last_pdf_bxdf, pdf_light);
                    radiance += throughput * bxdf_weight * emission;
                }
            }

            if (!is_frontfacing) {
                its.normal = -its.normal;
            }

            vec3 sample_dir = material->sample(its.normal, -ray.dir, bsdf_sample);
            auto sgeom_bxdf = get_shading_geom(its.normal, sample_dir, -ray.dir);

            Ray bxdf_ray = spawn_ray(its, sample_dir);

            f32 pdf = material->pdf(sgeom_bxdf);
            vec3 bxdf = material->eval(params.textures, its.uv);

            f32 light_sample = sampler->sample();
            auto sampled_light = sc->sample_lights(light_sample);
            if (sampled_light.has_value()) {
                // TODO: create a template for creating these vector samples...
                vec3 shape_rng =
                    vec3(sampler->sample(), sampler->sample(), sampler->sample());
                auto shape_sample = sc->geometry.sample_shape(
                    sampled_light.value().light.shape, its.pos, shape_rng);

                light_mis(its, ray, bxdf_ray, sampled_light.value(), shape_sample,
                          material, &radiance, throughput);
            }

            auto rr = russian_roulette(depth, rr_sample, throughput);
            if (!rr.has_value()) {
                break;
            }

            auto roulette_compensation = rr.value();

            throughput *=
                bxdf * sgeom_bxdf.cos_theta * (1.f / (pdf * roulette_compensation));

            ray = bxdf_ray;
            last_hit_pos = its.pos;
            last_pdf_bxdf = pdf;
            depth++;
        } else {
            // TODO: move into miss program to reduce divergence ?
            // Ray has escaped the scene
            if (!sc->has_envmap) {
                break;
            } else {
                const Envmap *envmap = &sc->envmap;
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
    auto *hit_data = reinterpret_cast<PtHitGroupData *>(optixGetSbtDataPointer());

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
