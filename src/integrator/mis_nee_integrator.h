#ifndef MIS_NEE_INTEGRATOR_H
#define MIS_NEE_INTEGRATOR_H

#include "../color/spectral_quantity.h"
#include "../geometry/ray.h"
#include "../materials/material.h"
#include "../settings.h"
#include "integrator.h"

struct LightSample;
struct Scene;
class SampledLambdas;
class Sampler;
class EmbreeAccel;
class IntegratorContext;

class MisNeeIntegrator final : public Integrator {
public:
    MisNeeIntegrator(const Settings &settings, IntegratorContext *ctx)
        : Integrator(ctx, settings) {}

    spectral
    estimate_radiance(Ray ray, Sampler &sampler, SampledLambdas &lambdas,
                      uvec2 &pixel) override;
};

#endif // MIS_NEE_INTEGRATOR_H
