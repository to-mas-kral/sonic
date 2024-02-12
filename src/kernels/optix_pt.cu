#include "optix_pt.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>

#include "../color/sampled_spectrum.h"
#include "../integrator/utils.h"
#include "../math/vecmath.h"
#include "../utils/basic_types.h"
#include "../utils/sampler.h"
#include "raygen.h"

extern "C" {
__constant__ PtParams params;
}

enum class HitType : u32 {
    Miss = 0,
    Triangle = 1,
    Sphere = 2,
};

static __forceinline__ __device__ void
set_payload_miss() {
    optixSetPayload_0(static_cast<u32>(HitType::Miss));
}

static __forceinline__ __device__ void
set_payload_hit_triangle(u32 prim_index, u32 mesh_id, float2 barycentrics) {
    optixSetPayload_0(static_cast<u32>(HitType::Triangle));
    optixSetPayload_1(prim_index);
    optixSetPayload_2(mesh_id);

    optixSetPayload_3(__float_as_uint(barycentrics.x));
    optixSetPayload_4(__float_as_uint(barycentrics.y));
}

static __forceinline__ __device__ void
set_payload_hit_sphere(u32 prim_index, f32 t) {
    optixSetPayload_0(static_cast<u32>(HitType::Sphere));
    optixSetPayload_1(prim_index);
    optixSetPayload_2(__float_as_uint(t));
}

static __forceinline__ __device__ Intersection
get_triangle_its(u32 bar_y, u32 bar_z, u32 triangle_index, u32 mesh_id) {
    f32 bar_y_f = __uint_as_float(bar_y);
    f32 bar_z_f = __uint_as_float(bar_z);
    vec3 bar = vec3(1.f - bar_y_f - bar_z_f, bar_y_f, bar_z_f);

    auto mesh_o = &params.meshes[mesh_id];

    auto geometry = &params.rc->scene.geometry;
    point3 *positions = &geometry->meshes.pos[mesh_o->pos_index];
    u32 *indices = &geometry->meshes.indices[mesh_o->indices_index];
    vec3 const *normals = geometry->meshes.normals.get_ptr_to(mesh_o->normals_index);
    vec2 const *uvs = geometry->meshes.uvs.get_ptr_to(mesh_o->uvs_index);

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
get_its(Scene *sc, u32 p1, u32 p2, u32 p3, u32 p4, HitType hit_type, Ray &ray) {
    if (hit_type == HitType::Triangle) {
        u32 prim_index = p1;
        u32 mesh_id = p2;
        u32 bar_y = p3;
        u32 bar_z = p4;

        return get_triangle_its(bar_y, bar_z, prim_index, mesh_id);
    } else {
        // Sphere
        u32 sphere_index = p1;
        f32 t = __uint_as_float(p2);

        Spheres &spheres = sc->geometry.spheres;
        point3 pos = ray.at(t);

        return get_sphere_its(sphere_index, spheres, pos);
    }
}

// Multisple Importance Sampling for lights
__device__ __forceinline__ spectral
light_mis(const Intersection &its, const Ray &traced_ray, const LightSample &light_sample,
          const norm_vec3 &geom_normal, const ShapeSample &shape_sample,
          const Material *material, const spectral &throughput,
          const SampledLambdas &lambdas) {
    point3 light_pos = shape_sample.pos;
    norm_vec3 pl = (light_pos - its.pos).normalized();
    f32 pl_mag_sq = (light_pos - its.pos).length_squared();
    f32 cos_light = vec3::dot(shape_sample.normal, -pl);

    auto sgeom_light = ShadingGeometry::make(its.normal, pl, -traced_ray.dir);

    // Quickly precheck if light is reachable
    if (sgeom_light.nowi > 0.f && cos_light > 0.f) {
        f32 pl_mag = (light_pos - its.pos).length();

        point3 ray_orig = offset_ray(its.pos, geom_normal);
        vec3 lrd = (light_pos - ray_orig).normalized();
        u32 did_hit = 1;
        // https://www.willusher.io/graphics/2019/09/06/faster-shadow-rays-on-rtx
        optixTrace(params.gas_handle, ray_orig.as_float3(), lrd.as_float3(), 0.f,
                   pl_mag - 0.001f, 0.0f, OptixVisibilityMask(255),
                   OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                       OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                   0, 1, 0, did_hit);

        if (!did_hit) {
            // Probability of sampling this light in terms of solid angle from the
            // probability distribution of the lights. Formula from
            // https://www.pbr-book.org/4ed/Radiometry,_Spectra,_and_Color/Working_with_Radiometric_Integrals#IntegralsoverArea
            f32 pdf_light = shape_sample.pdf * light_sample.pdf * (pl_mag_sq / cos_light);

            spectral bxdf_light =
                material->eval(sgeom_light, lambdas, params.textures, its.uv);
            f32 mat_pdf = material->pdf(sgeom_light, lambdas);

            f32 weight_light = mis_power_heuristic(pdf_light, mat_pdf);

            spectral light_emission = light_sample.light.emitter.emission(lambdas);

            return bxdf_light * sgeom_light.nowi * (1.f / pdf_light) * light_emission *
                   weight_light * throughput;
        }
    }

    return spectral::ZERO();
}

__device__ __forceinline__ spectral
bxdf_mis(Scene *sc, const spectral &throughput, const point3 &last_hit_pos,
         f32 last_pdf_bxdf, const Intersection &its, const spectral &emission) {
    norm_vec3 pl_norm = (its.pos - last_hit_pos).normalized();
    f32 pl_mag_sq = (its.pos - last_hit_pos).length_squared();
    f32 cos_light = vec3::dot(its.normal, -pl_norm);

    // last_pdf_bxdf is the probability of this light having been sampled
    // from the probability distribution of the BXDF of the *preceding*
    // hit.

    // TODO!!!: currently calculating the shape PDF by assuming pdf = 1. / area
    //  will have to change with non-uniform sampling !
    f32 light_area = sc->geometry.shape_area(sc->lights[its.light_id].shape);

    // pdf_light is the probability of this point being sampled from the
    // probability distribution of the lights.
    f32 pdf_light = sc->light_sampler.light_sample_pdf(its.light_id) * pl_mag_sq /
                    (light_area * cos_light);

    f32 bxdf_weight = mis_power_heuristic(last_pdf_bxdf, pdf_light);
    return throughput * emission * bxdf_weight;
}

extern "C" __global__ void
__raygen__rg() {
    const uint3 pixel = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    auto rc = params.rc;
    auto sc = &rc->scene;
    auto pixel_index = ((dim.y - 1U - pixel.y) * dim.x) + pixel.x;

    Sampler sampler{};
    sampler.init_frame(uvec2(pixel.x, pixel.y), uvec2(dim.x, dim.y), params.frame);

    auto cam_sample = sampler.sample2();

    auto ray =
        gen_ray(pixel.x, pixel.y, dim.x, dim.y, cam_sample, rc->cam, params.cam_to_world);

    SampledLambdas lambdas = SampledLambdas::new_sample_uniform(sampler.sample());

    spectral radiance = spectral::ZERO();
    spectral throughput = spectral::ONE();
    u32 depth = 1;
    f32 last_pdf_bxdf = 0.f;
    bool last_hit_specular = false;
    point3 last_hit_pos(0.f);

    while (true) {
        u32 p0, p1, p2, p3, p4;
        optixTrace(params.gas_handle, ray.o.as_float3(), vec_to_float3(ray.dir), 0.f,
                   1e16f, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                   0, 1, 0, p0, p1, p2, p3, p4);

        HitType hit_type{p0};
        if (hit_type != HitType::Miss) {
            auto bsdf_sample_rand = sampler.sample3();
            auto rr_sample = sampler.sample();
            Intersection its = get_its(sc, p1, p2, p3, p4, hit_type, ray);

            auto material = &params.materials[its.material_id];
            bool is_frontfacing = vec3::dot(-ray.dir, its.normal) >= 0.f;

            if (!is_frontfacing && !material->is_twosided) {
                break;
            }

            if (!is_frontfacing) {
                its.normal = -its.normal;
                its.geometric_normal = -its.geometric_normal;
            }

            if (its.has_light && is_frontfacing) {
                spectral emission = params.lights[its.light_id].emitter.emission(lambdas);

                if (params.integrator_type == IntegratorType::Naive || depth == 1 ||
                    last_hit_specular) {
                    // Primary ray hit, can't apply MIS...
                    radiance += throughput * emission;
                } else {
                    auto bxdf_mis_contrib = bxdf_mis(sc, throughput, last_hit_pos,
                                                     last_pdf_bxdf, its, emission);

                    radiance += bxdf_mis_contrib;
                }
            }

            // Do this before light sampling, because that "extends the path"
            if (params.max_depth > 0 && depth >= params.max_depth) {
                break;
            }

            last_hit_specular = material->is_dirac_delta();
            if (params.integrator_type != IntegratorType::Naive && !last_hit_specular) {
                f32 light_sample = sampler.sample();
                auto sampled_light = sc->sample_lights(light_sample);
                if (sampled_light.has_value()) {
                    auto shape_rng = sampler.sample3();
                    auto shape_sample = sc->geometry.sample_shape(
                        sampled_light.value().light.shape, its.pos, shape_rng);

                    auto light_mis_contrib =
                        light_mis(its, ray, sampled_light.value(), its.geometric_normal,
                                  shape_sample, material, throughput, lambdas);

                    radiance += light_mis_contrib;
                }
            }

            auto bsdf_sample_opt =
                material->sample(its.normal, -ray.dir, bsdf_sample_rand, lambdas,
                                 params.textures, its.uv, is_frontfacing);

            if (!bsdf_sample_opt.has_value()) {
                break;
            }
            auto bsdf_sample = bsdf_sample_opt.value();

            auto sgeom_bxdf = ShadingGeometry::make(its.normal, bsdf_sample.wi, -ray.dir);

            auto spawn_ray_normal =
                (bsdf_sample.did_refract) ? -its.geometric_normal : its.geometric_normal;
            Ray bxdf_ray = spawn_ray(its, spawn_ray_normal, bsdf_sample.wi);

            auto rr = russian_roulette(depth, rr_sample, throughput);
            if (!rr.has_value()) {
                break;
            }

            auto roulette_compensation = rr.value();
            throughput *= bsdf_sample.bsdf * sgeom_bxdf.cos_theta *
                          (1.f / (bsdf_sample.pdf * roulette_compensation));

            ray = bxdf_ray;
            last_hit_pos = its.pos;
            last_pdf_bxdf = bsdf_sample.pdf;
            depth++;

            if (depth == 1024) {
                printf("Specular infinite self-intersection path\n");
                // FIXME: specular infinite path caused by self-intersections
                break;
            }
        } else {
            // Ray has escaped the scene
            if (!sc->has_envmap) {
                break;
            } else {
                const Envmap *envmap = &sc->envmap;
                spectral envrad = envmap->get_ray_radiance(ray, lambdas);

                // TODO: do envmap sampling...
                /*if (depth == 1) {*/
                radiance += envrad;
                /*} else {
                    f32 env_pdf = envmap->pdf(ray.dir);
                    f32 env_weight = mis_power_heuristic(last_pdf_bxdf, env_pdf);

                    radiance += throughput * env_weight * envrad;
                }*/

                break;
            }
        }
    }

    params.fb->get_pixels()[pixel_index] += lambdas.to_xyz(radiance);
}

extern "C" __global__ void
__miss__ms() {
    set_payload_miss();
}

/*
 * From Nvidia docs:
 * "It is generally more efficient to have one hit shader handle multiple primitive
 * types (by switching on the value of optixGetPrimitiveType), rather than have several
 * hit shaders that implement the same ray behavior but differ only in the type of
 * geometry they expect."
 * */
extern "C" __global__ void
__closesthit__ch() {
    auto *hit_data = reinterpret_cast<PtHitGroupData *>(optixGetSbtDataPointer());

    auto type = optixGetPrimitiveType();

    if (type == OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
        // Triangles
        const float2 barycentrics = optixGetTriangleBarycentrics();
        const u32 prim_index = optixGetPrimitiveIndex();

        set_payload_hit_triangle(prim_index, hit_data->mesh.mesh_id, barycentrics);
    } else {
        // Spheres
        // CHECK: maybe add sphere_id to hit_data... would be safer if multiple
        //  sphere GASes are used in the future...
        const u32 sphere_index = optixGetSbtGASIndex();
        f32 t = optixGetRayTmax();

        set_payload_hit_sphere(sphere_index, t);
    }
}
