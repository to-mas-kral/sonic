#ifndef PT_EMITTER_H
#define PT_EMITTER_H

#include "math/math_utils.h"
#include "utils/basic_types.h"
#include "math/vecmath.h"

// Just a description of how a light emits light.
// More light sources can map onto the same emitter !
class Emitter {
public:
    explicit Emitter(const vec3 &emission) : _emission(emission) {}

    // For non-diffuse light this could depend on the incidence angle and so on...
    __device__ vec3
    emission() const {
        return _emission;
    }
    __host__ f32
    power() const {
        return (_emission.x + _emission.y + _emission.z) / 3.f;
    }

private:
    vec3 _emission;
};

#endif // PT_EMITTER_H
