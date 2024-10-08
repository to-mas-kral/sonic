#ifndef PT_EMITTER_H
#define PT_EMITTER_H

#include "../color/sampled_spectrum.h"
#include "../color/spectrum.h"
#include "../utils/basic_types.h"

// Just a description of how a light emits light.
// More light sources can map onto the same emitter !
class Emitter {
public:
    explicit
    Emitter(const Spectrum &emission, const bool twosided, const f32 scale)
        : twosided{twosided}, scale{scale}, _emission(emission) {}

    spectral
    emission(const SampledLambdas &lambdas) const;

    // Computes the average spectral power of the light source
    f32
    power() const;

    bool twosided = false;

private:
    f32 scale{1.f};
    Spectrum _emission;
};

#endif // PT_EMITTER_H
