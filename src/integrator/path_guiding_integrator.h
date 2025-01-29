#ifndef PATH_GUIDING_INTEGRATOR_H
#define PATH_GUIDING_INTEGRATOR_H

#include "../color/spectral_quantity.h"
#include "../geometry/ray.h"
#include "../integrator_context.h"
#include "../materials/material.h"
#include "../path_guiding/sd_tree.h"
#include "integrator.h"

struct PathVertices;
struct Scene;
class Sampler;
class EmbreeAccel;
class IntegratorContext;
struct LightSample;

class PathGuidingIntegrator final : public Integrator {
public:
    explicit PathGuidingIntegrator(const Settings &settings, IntegratorContext *ctx)
        : Integrator(ctx, settings), sd_tree(ctx->scene().bounds()) {}

    spectral
    estimate_radiance(Ray ray, Sampler &sampler, SampledLambdas &lambdas) override;

    void
    next_sample() override;

private:
    spectral
    radiance_rendering(Ray ray, Sampler &sampler, SampledLambdas &lambdas) const;

    spectral
    radiance_training(Ray ray, Sampler &sampler, SampledLambdas &lambdas);
    
    void
    add_radiance_contrib_learning(PathVertices &path_vertices, spectral &radiance,
                                  const spectral &path_contrib,
                                  const spectral &emission) const;

    SDTree sd_tree;

    bool training_phase{true};
    u32 training_iteration{0};
    u32 iteration_samples{0};
    u32 iteration_max_samples{1};
    u32 max_training_samples{128};

    bool do_intermediate_images{false};
};

#endif // PATH_GUIDING_INTEGRATOR_H
