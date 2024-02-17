#ifndef PT_MIS_NEE_INTEGRATOR_H
#define PT_MIS_NEE_INTEGRATOR_H

#include "../utils/sampler.h"
#include "hit_type.h"
#include "intersection.h"
#include "optix_pt.h"

// Multisple Importance Sampling for lights
__device__ __forceinline__ spectral
light_mis(const PtParams &params, const Intersection &its, const Ray &traced_ray,
          const LightSample &light_sample, const norm_vec3 &geom_normal,
          const ShapeSample &shape_sample, const Material *material,
          const spectral &throughput, const SampledLambdas &lambdas) {
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

__device__ spectral
integrator_mis_nee(const PtParams &params, Ray ray, Sampler &sampler,
                   const SampledLambdas &lambdas) {
    auto sc = &params.rc->scene;

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
            Intersection its = get_its(params.meshes, &params.rc->scene.geometry, sc, p1,
                                       p2, p3, p4, hit_type, ray);

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

                    auto light_mis_contrib = light_mis(
                        params, its, ray, sampled_light.value(), its.geometric_normal,
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
            Ray bxdf_ray = spawn_ray(its.pos, spawn_ray_normal, bsdf_sample.wi);

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
                radiance += envrad;
                break;
            }
        }
    }

    return radiance;
}

#endif // PT_MIS_NEE_INTEGRATOR_H
