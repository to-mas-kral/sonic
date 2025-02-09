#include "emitter.h"

spectral
Emitter::emission(const SampledLambdas &lambdas) const {
    return _emission.eval(lambdas) * scale;
}

f32
Emitter::emission(f32 lambda) const {
    return _emission.eval_single(lambda) * scale;
}

f32
Emitter::power() const {
    return _emission.power() * scale;
}
