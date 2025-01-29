#include "path_guiding_integrator.h"
#include "../io/image_writer.h"
#include "../utils/sampler.h"
#include "integrator.h"
#include "intersection.h"
#include "shading_frame.h"

void
PathGuidingIntegrator::next_sample() {
    sample++;

    if (!training_phase) {
        ctx->framebuf().num_samples++;
        return;
    }

    if (iteration_samples >= max_training_samples) {
        spdlog::info("Starting rendering");
        ctx->framebuf().reset();
        ctx->framebuf().num_samples++;
        training_phase = false;
        return;
    }

    if (iteration_samples >= iteration_max_samples) {
        spdlog::info("Resetting training iteration");
        iteration_max_samples *= 2;
        iteration_samples = 0;
        ctx->framebuf().reset();
        sd_tree.refine(training_iteration);
        training_iteration++;
    }

    if (do_intermediate_images && std::popcount(ctx->framebuf().num_samples) == 1) {
        ImageWriter::write_framebuffer(
            fmt::format("{}-i-{}-s-{}", ctx->attribs().film.filename.c_str(),
                        training_iteration, ctx->framebuf().num_samples),
            ctx->framebuf());
    }

    iteration_samples++;
    ctx->framebuf().num_samples++;
}

spectral
PathGuidingIntegrator::estimate_radiance(Ray ray, Sampler &sampler,
                                         SampledLambdas &lambdas) {

    if (!training_phase) {
        return radiance_training(ray, sampler, lambdas);
        // return radiance_rendering(ray, sampler, lambdas);
    } else {
        return radiance_training(ray, sampler, lambdas);
    }
}

spectral
PathGuidingIntegrator::radiance_rendering(Ray ray, Sampler &sampler,
                                          SampledLambdas &lambdas) const {
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
    add_to_sd_tree() {
        for (auto &vertex : path_vertices) {
            if (vertex.count != 0) {
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
PathGuidingIntegrator::add_radiance_contrib_learning(PathVertices &path_vertices,
                                                     spectral &radiance,
                                                     const spectral &path_contrib,
                                                     const spectral &emission) const {
    path_vertices.add_radiance_contrib(emission);
    add_radiance_contrib(radiance, path_contrib);
}

namespace {
struct DirectionalSample {
    bool did_refract = false;
    f32 pdf{0.F};
    f32 other_strategy_pdf{0.F};
    spectral bsdf{0.F};
    norm_vec3 wi_world_space;
    // Need to provide init for default init in integrator...
    ShadingFrame sframe{norm_vec3(), norm_vec3(), norm_vec3()};
};
} // namespace

spectral
PathGuidingIntegrator::radiance_training(Ray ray, Sampler &sampler,
                                         SampledLambdas &lambdas) {
    auto &sc = ctx->scene();
    auto &lights = ctx->scene().lights;
    auto &materials = ctx->scene().materials;
    auto max_depth = ctx->attribs().max_depth;

    spectral radiance = spectral::ZERO();
    u32 depth = 1;
    auto last_vertex = PathVertex{};
    auto path_vertices = PathVertices();

    while (true) {
        auto opt_its = ctx->accel().cast_ray(ray);
        if (!opt_its.has_value()) {
            if (sc.envmap) {
                const auto envrad = sc.envmap->get_ray_radiance(ray, lambdas);
                const auto path_contrib = last_vertex.throughput * envrad;
                add_radiance_contrib_learning(path_vertices, radiance, path_contrib,
                                              envrad);
            }
            break;
        }

        auto its = opt_its.value();

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
            const auto path_contrib = last_vertex.throughput * emission;
            add_radiance_contrib_learning(path_vertices, radiance, path_contrib,
                                          emission);
        }

        // Do this before light sampling, because that "extends the path"
        if (max_depth > 0 && depth >= max_depth) {
            break;
        }

        last_vertex.is_bxdf_delta = material->is_delta();

        const auto sampling_kind = sampler.sample();

        DirectionalSample sample;

        const bool do_mis = !material->is_delta() && training_iteration != 0;
        constexpr f32 bsdf_sampling_prob = 0.5F;

        if (material->is_delta() || training_iteration == 0 ||
            sampling_kind < bsdf_sampling_prob) {
            const auto bsdf_xi = sampler.sample3();
            // SD-tree cannot be used, sample using the BSDF
            const auto sframe = ShadingFrameIncomplete(its.normal);
            const auto bsdf_sample_opt = material->sample(
                sframe, -ray.dir(), bsdf_xi, lambdas, its.uv, is_frontfacing);
            if (!bsdf_sample_opt.has_value()) {
                break;
            }
            const auto bsdf_sample = bsdf_sample_opt.value();

            const auto &sframe_bsdf = bsdf_sample.sframe;
            if (sframe_bsdf.is_degenerate()) {
                break;
            }

            const auto wi_world_space =
                sframe_bsdf.from_local(sframe_bsdf.wi()).normalized();

            const auto sd_tree_pdf =
                do_mis
                    ? sd_tree.find_node(its.pos)->m_sampling_quadtree->pdf(wi_world_space)
                    : 0.F;

            sample =
                DirectionalSample(bsdf_sample.did_refract, bsdf_sample.pdf, sd_tree_pdf,
                                  bsdf_sample.bsdf, wi_world_space, bsdf_sample.sframe);
        } else {
            // Use SD-tree for sampling
            // TODO: make this function return the node so it doesn't have to be
            // fetched again later
            const auto sd_sample = sd_tree.sample(its.pos, sampler);
            const auto sd_sframe = ShadingFrame(its.normal, sd_sample.wi, -ray.dir());

            if (sd_sframe.is_degenerate()) {
                break;
            }

            if (sd_sframe.nowi() < 0.F && !material->is_translucent()) {
                break;
            }

            const auto bsdf = material->eval(sd_sframe, lambdas, its.uv);
            const auto bsdf_pdf =
                do_mis ? material->pdf(sd_sframe, lambdas, its.uv) : 0.F;

            const auto did_refract = sd_sframe.nowi() < 0.F;
            sample = DirectionalSample(did_refract, sd_sample.pdf, bsdf_pdf, bsdf,
                                       sd_sample.wi, sd_sframe);
        }

        if (do_mis) {
            // Do one-sample MIS between BSDF and sd-tree sampling
            const auto mis_weight =
                mis_power_heuristic(sample.pdf, sample.other_strategy_pdf);
            sample.bsdf *= mis_weight / bsdf_sampling_prob;
        }

        const auto spawn_ray_normal =
            sample.did_refract ? -its.geometric_normal : its.geometric_normal;
        const Ray bxdf_ray = spawn_ray(its.pos, spawn_ray_normal, sample.wi_world_space);

        const auto rr = russian_roulette(depth, rr_xi, last_vertex.throughput);
        if (!rr.has_value()) {
            break;
        }

        const auto roulette_compensation = rr.value();
        const auto pdf = sample.pdf * roulette_compensation;
        const auto vertex_throughput =
            sample.bsdf * sample.sframe.abs_nowi() * (1.F / pdf);
        last_vertex.throughput *= vertex_throughput;
        assert(!last_vertex.throughput.is_invalid());

        if (last_vertex.throughput.is_zero()) {
            break;
        }

        ray = bxdf_ray;
        last_vertex.pos = its.pos;
        last_vertex.pdf_bxdf = sample.pdf;
        depth++;

        if (training_phase) {
            path_vertices.add_vertex(
                PgVertex(sd_tree, last_vertex.pos, ray.dir(), vertex_throughput));
        }
    }

    if (training_phase) {
        path_vertices.add_to_sd_tree();
    }

    return radiance;
}
