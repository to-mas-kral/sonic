#include "emitter.h"

spectral
Emitter::emission(const SampledLambdas &lambdas) const {
    return _emission.eval(lambdas);
}

f32
Emitter::power() const {
    return _emission.power();
}
