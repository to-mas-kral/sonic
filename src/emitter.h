#ifndef PT_EMITTER_H
#define PT_EMITTER_H

#include "color/spectrum.h"
#include "color/sampled_spectrum.h"
#include "math/math_utils.h"
#include "math/vecmath.h"
#include "utils/basic_types.h"

// Just a description of how a light emits light.
// More light sources can map onto the same emitter !
class Emitter {
public:
    explicit Emitter(const RgbSpectrumIlluminant &emission) : _emission(emission) {}

    // For non-diffuse light this could depend on the incidence angle and so on...
    __device__ spectral
    emission(const SampledLambdas &lambdas) const {
        return _emission.eval(lambdas);
    }

    // TODO: this will be more tricky for texture illuminants...
    // Computes the average spectral power of the light source
    __host__ f32
    power() const {
        // Just use the rectangle rule...
        f32 sum = 0.f;
        constexpr u32 num_steps = 100;
        constexpr f32 lambda_min = static_cast<f32>(LAMBDA_MIN);
        constexpr f32 lambda_max = static_cast<f32>(LAMBDA_MAX);
        constexpr f32 h = (lambda_max - lambda_min) / static_cast<f32>(num_steps);
        for (u32 i = 0; i < num_steps; i++) {
            f32 lambda = lambda_min + (static_cast<f32>(i) * h) + h / 2.f;
            sum += _emission.eval_single(lambda);
        }

        f32 integral = sum * h;
        return integral / (lambda_max - lambda_min);
    }

private:
    RgbSpectrumIlluminant _emission;
};

#endif // PT_EMITTER_H
