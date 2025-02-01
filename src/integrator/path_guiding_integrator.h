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
    estimate_radiance(Ray ray, Sampler &sampler, SampledLambdas &lambdas,
                      uvec2 &pixel) override;

    void
    next_sample() override;

    std::optional<IterationProgressInfo>
    iter_progress_info() override;

private:
    spectral
    radiance_training(Ray ray, Sampler &sampler, SampledLambdas &lambdas, uvec2 &pixel);
    
    void
    add_radiance_contrib_learning(PathVertices &path_vertices, spectral &radiance,
                                  const spectral &path_contrib,
                                  const spectral &emission) const;

    std::tuple<spectral, spectral>
    light_mis_pg(const Intersection &its, const Ray &traced_ray,
              const LightSample &light_sample, const norm_vec3 &geom_normal,
              const Material *material, const spectral &throughput,
              const SampledLambdas &lambdas);
    
    SDTree sd_tree;

    bool training_phase{true};
    u32 training_iteration{0};
    u32 iteration_samples{0};
    u32 iteration_max_samples{1};
    u32 max_training_samples{64};

    bool do_intermediate_images{false};
    f32 bsdf_sampling_prob{0.5F};
};

#endif // PATH_GUIDING_INTEGRATOR_H
