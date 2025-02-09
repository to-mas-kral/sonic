#include "mis_nee_integrator.h"
#include "../math/samplers/sampler.h"
#include "integrator.h"
#include "intersection.h"
#include "shading_frame.h"

spectral
MisNeeIntegrator::estimate_radiance(Ray ray, Sampler &sampler, SampledLambdas &lambdas,
                                    uvec2 &pixel) {
    auto &sc = ctx->scene();
    auto &lights = ctx->scene().lights;
    auto &materials = ctx->scene().materials;
    auto max_depth = ctx->attribs().max_depth;

    spectral radiance = spectral::ZERO();
    u32 depth = 1;
    auto last_vertex = PathVertex{};

    while (true) {
        auto opt_its = ctx->accel().cast_ray(ray);
        if (!opt_its.has_value()) {
            if (sc.envmap) {
                const auto envrad = sc.envmap->get_ray_radiance(ray, lambdas);

                if (depth == 1 || last_vertex.is_bxdf_delta ||
                    settings.integrator_type == IntegratorType::Naive) {
                    // straight miss... can't do MIS
                    const auto contrib = last_vertex.throughput * envrad;
                    add_radiance_contrib(radiance, contrib);
                } else {
                    const f32 pdf_envmap =
                        sc.light_sampler.light_sample_pdf(sc.envmap->light_id()) *
                        sc.envmap->pdf(ray.dir());
                    const f32 bxdf_weight =
                        mis_power_heuristic(last_vertex.pdf_bxdf, pdf_envmap);

                    // pdf is already contained in the throughput
                    const auto contrib = last_vertex.throughput * envrad * bxdf_weight;
                    add_radiance_contrib(radiance, contrib);
                }
            }
            break;
        }

        auto its = opt_its.value();

        if (depth == 1) {
            record_aovs(pixel, its);
        }

        const auto bsdf_xi = sampler.sample3();
        const auto rr_xi = sampler.sample();

        auto *const material = &materials[its.material_id.inner];
        const auto is_frontfacing = vec3::dot(-ray.dir(), its.normal) >= 0.F;

        if (!is_frontfacing && !material->is_twosided) {
            break;
        }

        if (!is_frontfacing) {
            its.normal = -its.normal;
            its.geometric_normal = -its.geometric_normal;
        }

        if (its.has_light && is_frontfacing) {
            const spectral emission = lights[its.light_id].emission(lambdas);

            if (settings.integrator_type == IntegratorType::Naive || depth == 1 ||
                last_vertex.is_bxdf_delta) {
                // Primary ray hit, can't apply MIS...
                const auto contrib = last_vertex.throughput * emission;
                add_radiance_contrib(radiance, contrib);
            } else {
                auto bxdf_mis_contrib = bxdf_mis(last_vertex.throughput, last_vertex.pos,
                                                 last_vertex.pdf_bxdf, its, emission);

                add_radiance_contrib(radiance, bxdf_mis_contrib);
            }
        }

        // Do this before light sampling, because that "extends the path"
        if (max_depth > 0 && depth >= max_depth) {
            break;
        }

        const f32 light_xi = sampler.sample();
        const auto shape_xi = sampler.sample3();

        last_vertex.is_bxdf_delta = material->is_delta();
        if (settings.integrator_type != IntegratorType::Naive &&
            !last_vertex.is_bxdf_delta) {
            const auto sampled_light = sc.sample_lights(light_xi, shape_xi, lambdas, its);
            if (sampled_light.has_value()) {
                auto light_mis_contrib =
                    light_mis(its, ray, sampled_light.value(), its.geometric_normal,
                              material, last_vertex.throughput, lambdas);

                add_radiance_contrib(radiance, light_mis_contrib);
            }
        }

        const auto sframe = ShadingFrameIncomplete(its.normal);
        const auto bsdf_sample_opt = material->sample(sframe, -ray.dir(), bsdf_xi,
                                                      lambdas, its.uv, is_frontfacing);

        if (!bsdf_sample_opt.has_value()) {
            break;
        }
        const auto bsdf_sample = bsdf_sample_opt.value();

        const auto &sframe_bsdf = bsdf_sample.sframe;
        if (sframe_bsdf.is_degenerate()) {
            break;
        }

        const auto spawn_ray_normal =
            (bsdf_sample.did_refract) ? -its.geometric_normal : its.geometric_normal;
        const Ray bxdf_ray =
            spawn_ray(its.pos, spawn_ray_normal,
                      sframe_bsdf.from_local(sframe_bsdf.wi()).normalized());

        const auto rr = russian_roulette(depth, rr_xi, last_vertex.throughput);
        if (!rr.has_value()) {
            break;
        }

        const auto roulette_compensation = rr.value();
        last_vertex.throughput *= bsdf_sample.bsdf * sframe_bsdf.abs_nowi() *
                                  (1.F / (bsdf_sample.pdf * roulette_compensation));
        assert(!last_vertex.throughput.is_invalid());

        if (last_vertex.throughput.is_zero()) {
            break;
        }

        ray = bxdf_ray;
        last_vertex.pos = its.pos;
        last_vertex.pdf_bxdf = bsdf_sample.pdf;
        depth++;
    }

    return radiance;
}
