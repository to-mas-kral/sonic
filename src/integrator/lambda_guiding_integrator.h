#ifndef LAMBDA_GUIDING_INTEGRATOR_H
#define LAMBDA_GUIDING_INTEGRATOR_H

#include "../geometry/ray.h"
#include "../materials/material.h"
#include "../path_guiding/sc_tree.h"
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
        : Integrator(ctx, settings), lg_tree(ctx->scene().materials.size()) {}

    spectral
    estimate_radiance(Ray ray, Sampler &sampler, SampledLambdas &lambdas,
                      uvec2 &pixel) override;

    void
    advance_sample() override;

    void
    reset_iteration() override;

    std::optional<IterationProgressInfo>
    iter_progress_info() const override;

    std::optional<LgTree>
    get_lg_tree() const override {
        return lg_tree;
    }

protected:
    void
    record_aovs(const uvec2 &pixel, const Intersection &its) override;

private:
    LgTree lg_tree;

    bool training_phase{true};
    u32 training_iteration{0};
    u32 iteration_samples{0};
    u32 iteration_max_samples{1};
    u32 max_training_samples{4};
};

#endif // LAMBDA_GUIDING_INTEGRATOR_H
