#ifndef PATH_GUIDING_INTEGRATOR_H
#define PATH_GUIDING_INTEGRATOR_H

#include "../color/spectral_quantity.h"
#include "../geometry/ray.h"
#include "../materials/material.h"
#include "../path_guiding/sd_tree.h"
#include "../render_context.h"
#include "../settings.h"
#include "intersection.h"

struct PathVertices;
struct Scene;
class Sampler;
class EmbreeDevice;
class RenderContext;
struct LightSample;

class PathGuidingIntegrator {
public:
    PathGuidingIntegrator(const Settings &settings, RenderContext *rc,
                          EmbreeDevice *device)
        : sd_tree(rc->scene.bounds()), rc{rc}, settings{settings}, device{device} {}

    spectral
    radiance_rendering(Ray ray, Sampler &sampler, SampledLambdas &lambdas) const;

    spectral
    radiance_training(Ray ray, Sampler &sampler, SampledLambdas &lambdas);

private:
    spectral
    light_mis(const Intersection &its, const Ray &traced_ray,
              const LightSample &light_sample, const norm_vec3 &geom_normal,
              const Material *material, const spectral &throughput,
              const SampledLambdas &lambdas) const;

    spectral
    bxdf_mis(const Scene &sc, const spectral &throughput, const point3 &last_hit_pos,
             f32 last_pdf_bxdf, const Intersection &its, const spectral &emission) const;

    void
    add_radiance_contrib(spectral &radiance, const spectral &contrib) const;

    void
    add_radiance_contrib_learning(PathVertices &path_vertices, spectral &radiance,
                                  const spectral &path_contrib,
                                  const spectral &emission) const;

    SDTree sd_tree;

    RenderContext *rc;
    Settings settings;
    EmbreeDevice *device;
};

#endif // PATH_GUIDING_INTEGRATOR_H
