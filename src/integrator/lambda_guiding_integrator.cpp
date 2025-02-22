#include "lambda_guiding_integrator.h"

#include "../math/samplers/sampler.h"
#include "../utils/colormaps.h"
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
        sd_tree.refine(training_iteration);
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

struct PgVertex {
    PgVertex(SDTree &sd_tree, const point3 &pos, const norm_vec3 &wi,
             const spectral &throughput)
        : throughput(throughput), wi(wi), tree_node(sd_tree.find_node(pos)) {}

    void
    add_radiance(const spectral &radiance_contrib) {
        radiance += throughput * radiance_contrib;
        count++;
    }

    spectral radiance{0.F};
    spectral throughput;
    u32 count{0};
    norm_vec3 wi;

    SDTreeNode *tree_node;
};

struct PathVertices {
    void
    add_vertex(const PgVertex &vertex) {
        for (auto &existing_vertex : path_vertices) {
            existing_vertex.throughput *= vertex.throughput;
        }

        path_vertices.push_back(vertex);
    }

    void
    add_radiance_contrib(const spectral &radiance_contrib) {
        for (auto &vertex : path_vertices) {
            vertex.add_radiance(radiance_contrib);
        }
    }

    void
    add_to_sd_tree(const MaterialId mat_id, const SampledLambdas &lambdas,
                   bool record_lambdas = false) {
        for (auto &vertex : path_vertices) {
            if (vertex.count == 0) {
                continue;
            }

            if (record_lambdas) {
                vertex.tree_node->record_bulk(vertex.radiance, lambdas, vertex.wi, mat_id,
                                              vertex.count);
            } else {
                vertex.tree_node->record_bulk(vertex.radiance, vertex.wi, vertex.count);
            }
        }
    }

    bool
    is_empty() const {
        return path_vertices.empty();
    }

    const PgVertex &
    last_vertex() const {
        assert(!path_vertices.empty());
        return path_vertices.back();
    }

private:
    std::vector<PgVertex> path_vertices{};
};

void
LambdaGuidingIntegrator::record_aovs(const uvec2 &pixel, const Intersection &its) {
    Integrator::record_aovs(pixel, its);

    if (iteration_samples == 0) {
        // TOOD: refactor - make sd_tree.find_node const.
        const auto id = sd_tree.find_node_id(its.pos);
        const auto color = sonic::colormap(id);

        ctx->framebuf().add_aov(pixel, "S-Tree ID",
                                color * std::max(its.normal.max_component(),
                                                 std::abs(its.normal.min_component())));
    }
}

void
LambdaGuidingIntegrator::add_radiance_contrib_learning(PathVertices &path_vertices,
                                                       spectral &radiance,
                                                       const spectral &path_contrib,
                                                       const spectral &emission) const {
    path_vertices.add_radiance_contrib(emission);
    add_radiance_contrib(radiance, path_contrib);
}

/// Returns the contribution and emission.
std::tuple<spectral, spectral>
LambdaGuidingIntegrator::light_mis_pg(const Intersection &its, const Ray &traced_ray,
                                      const LightSample &light_sample,
                                      const norm_vec3 &geom_normal,
                                      const Material *material,
                                      const spectral &throughput,
                                      const SampledLambdas &lambdas) const {
    const point3 light_pos = light_sample.pos;
    const norm_vec3 pl = (light_pos - its.pos).normalized();
    // TODO: move this into light sampling itself
    const f32 cos_light = vec3::dot(light_sample.normal, -pl);

    const auto sframe_light = ShadingFrame(its.normal, pl, -traced_ray.dir());
    if (sframe_light.is_degenerate()) {
        return {spectral::ZERO(), spectral::ZERO()};
    }

    // Quickly precheck if light is reachable
    if (sframe_light.nowi() > 0.F && cos_light > 0.F) {
        const point3 ray_orig = offset_ray(its.pos, geom_normal);
        const spectral bxdf_light = material->eval(sframe_light, lambdas, its.uv);
        // eval bxdf and check if it's 0 to potentially avoid the raycast
        if (bxdf_light.is_zero()) {
            return {spectral::ZERO(), spectral::ZERO()};
        }

        if (ctx->accel().is_visible(ray_orig, light_pos)) {
            const auto mat_pdf = material->pdf(sframe_light, lambdas, its.uv);

            const f32 weight_light = mis_power_heuristic(light_sample.pdf, mat_pdf);

            const auto contrib = bxdf_light * sframe_light.abs_nowi() *
                                 (1.F / light_sample.pdf) * light_sample.emission *
                                 weight_light * throughput;

            assert(!contrib.is_invalid());
            return {contrib, light_sample.emission};
        }
    }

    return {spectral::ZERO(), spectral::ZERO()};
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
    auto path_vertices = PathVertices();

    MaterialId first_mat_id{0};

    while (true) {
        auto opt_its = ctx->accel().cast_ray(ray);
        if (!opt_its.has_value()) {
            if (sc.envmap) {
                const auto envrad = sc.envmap->get_ray_radiance(ray, lambdas);

                if (depth == 1 || last_vertex.is_bxdf_delta ||
                    settings.integrator_type == IntegratorType::Naive) {
                    // straight miss... can't do MIS
                    const auto contrib = last_vertex.throughput * envrad;
                    add_radiance_contrib_learning(path_vertices, radiance, contrib,
                                                  envrad);
                } else {
                    const f32 pdf_envmap =
                        sc.light_sampler.light_sample_pdf(sc.envmap->light_id()) *
                        sc.envmap->pdf(ray.dir());
                    const f32 bxdf_weight =
                        mis_power_heuristic(last_vertex.pdf_bxdf, pdf_envmap);

                    // pdf is already contained in the throughput
                    const auto contrib = last_vertex.throughput * envrad * bxdf_weight;
                    add_radiance_contrib_learning(path_vertices, radiance, contrib,
                                                  envrad);
                }
            }
            break;
        }

        auto its = opt_its.value();

        if (depth == 1) {
            record_aovs(pixel, its);

            first_mat_id = its.material_id;

            if (training_phase && training_iteration > 4) {
                // subequent iteration
                const auto &node = sd_tree.find_node(its.pos);

                //
                // MIS sampling
                //
                constexpr f32 guiding_prob = 0.5F;
                constexpr f32 other_prob = 1.F - guiding_prob;
                const auto kind = sampler.sample();
                spectral other_pdf;
                if (kind < guiding_prob) {
                    // Sample guiding tree
                    lambdas = node->m_sampling_binarytrees->sample(first_mat_id, sampler);

                    for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
                        other_pdf[i] = 0.003939804229F /
                                       sqr(std::coshf(0.0072F * (lambdas[i] - 538.F)));
                    }

                    for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
                        const auto mis_weight =
                            mis_power_heuristic(lambdas.pdfs[i], other_pdf[i]);
                        lambdas.weights[i] = mis_weight / guiding_prob;
                    }
                } else {
                    // Sample regular
                    lambdas = SampledLambdas::new_sample_importance(sampler);
                    for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
                        other_pdf[i] =
                            node->m_sampling_binarytrees->pdf(first_mat_id, lambdas[i]);
                    }

                    for (int i = 0; i < N_SPECTRUM_SAMPLES; ++i) {
                        const auto mis_weight =
                            mis_power_heuristic(lambdas.pdfs[i], other_pdf[i]);
                        lambdas.weights[i] = mis_weight / other_prob;
                    }
                }
            } else if (!training_phase) {
                const auto &node = sd_tree.find_node(its.pos);
                lambdas = node->m_sampling_binarytrees->sample(first_mat_id, sampler);
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
                add_radiance_contrib_learning(path_vertices, radiance, contrib, emission);
            } else {
                auto bxdf_mis_contrib = bxdf_mis(last_vertex.throughput, last_vertex.pos,
                                                 last_vertex.pdf_bxdf, its, emission);

                add_radiance_contrib_learning(path_vertices, radiance, bxdf_mis_contrib,
                                              emission);
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
                auto [light_mis_contrib, emission] =
                    light_mis_pg(its, ray, sampled_light.value(), its.geometric_normal,
                                 material, last_vertex.throughput, lambdas);

                add_radiance_contrib_learning(path_vertices, radiance, light_mis_contrib,
                                              emission);
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

        if (training_phase) {
            path_vertices.add_vertex(
                PgVertex(sd_tree, last_vertex.pos, ray.dir(), vertex_throughput));
        }
    }

    if (training_phase) {
        path_vertices.add_to_sd_tree(first_mat_id, lambdas, training_iteration > 3);
    }

    return radiance;
}
