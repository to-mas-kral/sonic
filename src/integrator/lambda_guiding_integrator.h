#ifndef LAMBDA_GUIDING_INTEGRATOR_H
#define LAMBDA_GUIDING_INTEGRATOR_H

#include "../geometry/ray.h"
#include "../materials/material.h"
#include "../path_guiding/sd_tree.h"
#include "../settings.h"
#include "../spectrum/spectral_quantity.h"
#include "integrator.h"

struct PathVertices;
struct LightSample;
struct Scene;
class SampledLambdas;
class Sampler;
class EmbreeAccel;
class IntegratorContext;

class LambdaGuidingIntegrator final : public Integrator {
public:
    LambdaGuidingIntegrator(const Settings &settings, IntegratorContext *ctx)
        : Integrator(ctx, settings), sd_tree(ctx->scene().bounds()) {}

    spectral
    estimate_radiance(Ray ray, Sampler &sampler, SampledLambdas &lambdas,
                      uvec2 &pixel) override;

    void
    advance_sample() override;

    void
    reset_iteration() override;

    std::optional<IterationProgressInfo>
    iter_progress_info() const override;

    std::optional<SDTree>
    get_sd_tree() const override {
        return sd_tree;
    }

private:
    void
    add_radiance_contrib_learning(PathVertices &path_vertices, spectral &radiance,
                                  const spectral &path_contrib,
                                  const spectral &emission) const;

    std::tuple<spectral, spectral>
    light_mis_pg(const Intersection &its, const Ray &traced_ray,
                 const LightSample &light_sample, const norm_vec3 &geom_normal,
                 const Material *material, const spectral &throughput,
                 const SampledLambdas &lambdas) const;

    SDTree sd_tree;

    bool training_phase{true};
    u32 training_iteration{0};
    u32 iteration_samples{0};
    u32 iteration_max_samples{1};
    u32 max_training_samples{64};
};

#endif // LAMBDA_GUIDING_INTEGRATOR_H
