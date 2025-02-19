#include "mis_nee_integrator.h"
#include "../utils/sampler.h"
#include "integrator.h"
#include "intersection.h"
#include "shading_frame.h"

// Multiple Importance Sampling for lights
spectral
MisNeeIntegrator::light_mis(const Intersection &its, const Ray &traced_ray,
                            const LightSample &light_sample, const norm_vec3 &geom_normal,
                            const Material *material, const spectral &throughput,
                            const SampledLambdas &lambdas) const {
    const point3 light_pos = light_sample.pos;
    const norm_vec3 pl = (light_pos - its.pos).normalized();
    // TODO: move this into light sampling itself
    const f32 cos_light = vec3::dot(light_sample.normal, -pl);

    const auto sframe_light = ShadingFrame(its.normal, pl, -traced_ray.dir());
    if (sframe_light.is_degenerate()) {
        return spectral::ZERO();
    }

    // Quickly precheck if light is reachable
    if (sframe_light.nowi() > 0.F && cos_light > 0.F) {
        const point3 ray_orig = offset_ray(its.pos, geom_normal);
        const spectral bxdf_light = material->eval(sframe_light, lambdas, its.uv);
        // eval bxdf and check if it's 0 to potentially avoid the raycast
        if (bxdf_light.is_zero()) {
            return spectral::ZERO();
        }

        if (device->is_visible(ray_orig, light_pos)) {
            const f32 mat_pdf = material->pdf(sframe_light, lambdas, its.uv);

            const f32 weight_light = mis_power_heuristic(light_sample.pdf, mat_pdf);

            const auto contrib = bxdf_light * sframe_light.abs_nowi() *
                                 (1.F / light_sample.pdf) * light_sample.emission *
                                 weight_light * throughput;

            assert(!contrib.is_invalid());
            return contrib;
        }
    }

    return spectral::ZERO();
}

spectral
MisNeeIntegrator::bxdf_mis(const Scene &sc, const spectral &throughput,
                           const point3 &last_hit_pos, const f32 last_pdf_bxdf,
                           const Intersection &its, const spectral &emission) const {
    const norm_vec3 pl_norm = (its.pos - last_hit_pos).normalized();
    const f32 pl_mag_sq = (its.pos - last_hit_pos).length_squared();
    const f32 cos_light = vec3::dot(its.normal, -pl_norm);

    // last_pdf_bxdf is the probability of this light having been sampled
    // from the probability distribution of the BXDF of the *preceding*
    // hit.

    // TODO!!!: currently calculating the light PDF by assuming pdf = 1. / area
    //  will have to change with non-uniform sampling !
    const f32 shape_pdf = 1.F / sc.lights[its.light_id].area(sc.geometry_container);

    // pdf_light is the probability of this point being sampled from the
    // probability distribution of the lights.
    const f32 pdf_light = sc.light_sampler.light_sample_pdf(its.light_id) * pl_mag_sq *
                          shape_pdf / (cos_light);

    const f32 bxdf_weight = mis_power_heuristic(last_pdf_bxdf, pdf_light);
    // pdf is already contained in the throughput
    const auto contrib = throughput * emission * bxdf_weight;
    return contrib;
}

void
MisNeeIntegrator::add_radiance_contrib(spectral &radiance,
                                       const spectral &contrib) const {
    if (contrib.max_component() > 10000.F) {
        spdlog::warn("potential firefly");
    }
    assert(!contrib.is_invalid());
    assert(!radiance.is_invalid());
    radiance += contrib;
}

struct Vertex {
    spectral throughput{spectral::ONE()};
    f32 pdf_bxdf{0.F};
    bool is_bxdf_delta{false};
    point3 pos{0.F};
};

spectral
MisNeeIntegrator::radiance(Ray ray, Sampler &sampler, SampledLambdas &lambdas) const {
    auto &sc = rc->scene;
    auto &lights = rc->scene.lights;
    auto &materials = rc->scene.materials;
    auto max_depth = rc->attribs.max_depth;

    spectral radiance = spectral::ZERO();
    u32 depth = 1;
    auto last_vertex = Vertex{};

#ifndef NDEBUG
    std::vector<Vertex> path_vertices{};
#endif

    while (true) {
        auto opt_its = device->cast_ray(ray);
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
                auto bxdf_mis_contrib =
                    bxdf_mis(sc, last_vertex.throughput, last_vertex.pos,
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

#ifndef NDEBUG
        path_vertices.emplace_back(last_vertex);
#endif

        if (depth == 1024) {
            fmt::println("infinite self-intersection path");
            break;
        }
    }

    return radiance;
}
