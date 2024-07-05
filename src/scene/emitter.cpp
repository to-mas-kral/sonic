#include "emitter.h"

spectral
Emitter::emission(const SampledLambdas &lambdas) const {
    return _emission.eval(lambdas) * scale;
}

f32
Emitter::power() const {
    return _emission.power() * scale;
}
