#ifndef MIS_NEE_INTEGRATOR_H
#define MIS_NEE_INTEGRATOR_H

#include "../color/spectral_quantity.h"
#include "../geometry/ray.h"
#include "../materials/material.h"
#include "../settings.h"
#include "intersection.h"

struct LightSample;
struct Scene;
class SampledLambdas;
class Sampler;
class EmbreeDevice;
class RenderContext;

class MisNeeIntegrator {
public:
    MisNeeIntegrator(const Settings &settings, RenderContext *rc, EmbreeDevice *device)
        : rc{rc}, settings{settings}, device{device} {}

    spectral
    radiance(Ray ray, Sampler &sampler, SampledLambdas &lambdas) const;

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

    RenderContext *rc;
    Settings settings;
    EmbreeDevice *device;
};

#endif // MIS_NEE_INTEGRATOR_H
