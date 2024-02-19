#ifndef PT_EMITTER_H
#define PT_EMITTER_H

#include "../color/sampled_spectrum.h"
#include "../color/spectrum.h"
#include "../utils/basic_types.h"

// Just a description of how a light emits light.
// More light sources can map onto the same emitter !
class Emitter {
public:
    explicit Emitter(const RgbSpectrumIlluminant &emission) : _emission(emission) {}

    spectral
    emission(const SampledLambdas &lambdas) const;

    // Computes the average spectral power of the light source
    f32
    power() const;

private:
    RgbSpectrumIlluminant _emission;
};

#endif // PT_EMITTER_H
