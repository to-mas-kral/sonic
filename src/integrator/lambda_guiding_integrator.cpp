#include "lambda_guiding_integrator.h"

#include "../math/samplers/sampler.h"
#include "../utils/float_comparison.h"
#include "integrator.h"
#include "intersection.h"
#include "shading_frame.h"

void
LambdaGuidingIntegrator::advance_sample() {
    sample++;
    iteration_samples++;
    ctx->framebuf().num_samples++;
}

void
LambdaGuidingIntegrator::reset_iteration() {
    if (!training_phase) {
        return;
    }

    if (iteration_samples >= max_training_samples) {
        spdlog::info("Starting rendering");
        ctx->framebuf().reset();
        iteration_samples = 0;
        training_phase = false;
    }

    if (iteration_samples >= iteration_max_samples) {
        spdlog::info("Resetting training iteration");
        iteration_max_samples *= 2;
        ctx->framebuf().reset();
        iteration_samples = 0;
        lg_tree.refine();
        training_iteration++;
    }
}

std::optional<IterationProgressInfo>
LambdaGuidingIntegrator::iter_progress_info() const {
    if (training_phase) {
        return IterationProgressInfo{
            .samples_max = iteration_max_samples,
            .samples_done = iteration_samples,
        };
    } else {
        assert(std::popcount(max_training_samples) == 1);
        return IterationProgressInfo{
            .samples_max = settings.spp - (2 * max_training_samples - 1),
            .samples_done = iteration_samples,
        };
    }
}

void
LambdaGuidingIntegrator::record_aovs(const uvec2 &pixel, const Intersection &its) {
    Integrator::record_aovs(pixel, its);

    if (iteration_samples == 0) {
        const auto reservoir =
            reinterpret_cast<u64>(&lg_tree.find_reservoir(its.material_id, pixel));
        const auto hash = hash_buffer(&reservoir, sizeof(reservoir));
        ctx->framebuf().add_aov(pixel, "Reservoir", sonic::colormap(hash));
    }
}

spectral
LambdaGuidingIntegrator::estimate_radiance(Ray ray, Sampler &sampler,
                                           SampledLambdas &lambdas, uvec2 &pixel) {
    auto &sc = ctx->scene();
    auto &lights = ctx->scene().lights;
    auto &materials = ctx->scene().materials;
    auto max_depth = ctx->attribs().max_depth;

    spectral radiance = spectral::ZERO();
    u32 depth = 1;
    auto last_vertex = PathVertex{};

    MaterialId first_mat_id{0};
    Reservoir *reservoir = nullptr;

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
            first_mat_id = its.material_id;
            record_aovs(pixel, its);

            reservoir = &lg_tree.find_reservoir(first_mat_id, pixel);

            if (training_phase && training_iteration > 4) {
                // subequent iteration

                //
                // MIS sampling
                //
                constexpr f32 guiding_prob = 0.5F;
                constexpr f32 other_prob = 1.F - guiding_prob;
                const auto kind = sampler.sample();
                spectral other_pdf;
                if (kind < guiding_prob) {
                    // Sample guiding tree
                    lambdas = reservoir->sample(sampler.sample());

                    for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
                        other_pdf[i] = SampledLambdas::pdf_visual_importance(lambdas[i]);
                    }

                    for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
                        const auto mis_weight =
                            mis_power_heuristic(lambdas.pdfs[i], other_pdf[i]);
                        lambdas.weights[i] = mis_weight / guiding_prob;
                    }
                } else {
                    // Sample regular
                    lambdas = SampledLambdas::sample_visual_importance(sampler.sample());
                    for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
                        other_pdf[i] = reservoir->pdf(lambdas[i]);
                    }

                    for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
                        const auto mis_weight =
                            mis_power_heuristic(lambdas.pdfs[i], other_pdf[i]);
                        lambdas.weights[i] = mis_weight / other_prob;
                    }
                }
            } else if (!training_phase) {
                lambdas = reservoir->sample(sampler.sample());
            }
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
        const auto pdf = bsdf_sample.pdf * roulette_compensation;
        const auto vertex_throughput =
            bsdf_sample.bsdf * bsdf_sample.sframe.abs_nowi() * (1.F / pdf);
        last_vertex.throughput *= vertex_throughput;
        assert(!last_vertex.throughput.is_invalid());

        if (last_vertex.throughput.is_zero()) {
            break;
        }

        ray = bxdf_ray;
        last_vertex.pos = its.pos;
        last_vertex.pdf_bxdf = bsdf_sample.pdf;
        depth++;
    }

    if (reservoir != nullptr && training_iteration > 0) {
        reservoir->recording_binary_tree->record(lambdas, radiance);
    }

    return radiance;
}
