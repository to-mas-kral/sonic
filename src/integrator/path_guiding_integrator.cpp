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
        iteration_samples++;
        return;
    }

    if (iteration_samples >= max_training_samples) {
        spdlog::info("Starting rendering");
        ctx->framebuf().reset();
        ctx->framebuf().num_samples++;
        training_phase = false;
        iteration_samples = 0;
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

std::optional<IterationProgressInfo>
PathGuidingIntegrator::iter_progress_info() {
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

spectral
PathGuidingIntegrator::estimate_radiance(Ray ray, Sampler &sampler,
                                         SampledLambdas &lambdas, uvec2 &pixel) {
    return radiance_training(ray, sampler, lambdas, pixel);
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

/// Returns the contribution and emission.
std::tuple<spectral, spectral>
PathGuidingIntegrator::light_mis_pg(const Intersection &its, const Ray &traced_ray,
                                    const LightSample &light_sample,
                                    const norm_vec3 &geom_normal,
                                    const Material *material, const spectral &throughput,
                                    const SampledLambdas &lambdas) {
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
            const auto *const sd_tree_node = sd_tree.find_node(its.pos);
            const auto guide_pdf = sd_tree_node->m_sampling_quadtree->pdf(pl);

            const f32 directional_pdf =
                bsdf_sampling_prob * mat_pdf + (1.F - bsdf_sampling_prob) * guide_pdf;

            const f32 weight_light =
                mis_power_heuristic(light_sample.pdf, directional_pdf);

            const auto contrib = bxdf_light * sframe_light.abs_nowi() *
                                 (1.F / light_sample.pdf) * light_sample.emission *
                                 weight_light * throughput;

            assert(!contrib.is_invalid());
            return {contrib, light_sample.emission};
        }
    }

    return {spectral::ZERO(), spectral::ZERO()};
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
                                         SampledLambdas &lambdas, uvec2 &pixel) {
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

                // TODO: fix Path Guiding + NEE
                if (true || depth == 1 || last_vertex.is_bxdf_delta) {
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
        }

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

            // TODO: fix Path Guiding + NEE
            if (true || depth == 1 || last_vertex.is_bxdf_delta) {
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

        last_vertex.is_bxdf_delta = material->is_delta();

        const f32 light_xi = sampler.sample();
        const auto shape_xi = sampler.sample3();

        // TODO: fix Path Guiding + NEE
        if (false && !last_vertex.is_bxdf_delta) {
            const auto sampled_light = sc.sample_lights(light_xi, shape_xi, lambdas, its);
            if (sampled_light.has_value()) {
                auto [light_mis_contrib, emission] =
                    light_mis_pg(its, ray, sampled_light.value(), its.geometric_normal,
                                 material, last_vertex.throughput, lambdas);

                // Don't add NEE contrib to guiding tree as we want mostly indirect
                // illumination there.
                /*add_radiance_contrib_learning(path_vertices, radiance,
                   light_mis_contrib, emission);*/
                add_radiance_contrib(radiance, light_mis_contrib);
            }
        }

        const auto sampling_kind = sampler.sample();

        DirectionalSample sample;

        const bool do_mis = !material->is_delta() && training_iteration != 0;

        if (material->is_delta() || training_iteration == 0 ||
            sampling_kind < bsdf_sampling_prob) {
            const auto bsdf_xi = sampler.sample3();
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

            const auto guide_pdf =
                do_mis
                    ? sd_tree.find_node(its.pos)->m_sampling_quadtree->pdf(wi_world_space)
                    : 0.F;

            sample =
                DirectionalSample(bsdf_sample.did_refract, bsdf_sample.pdf, guide_pdf,
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
