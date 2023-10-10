#ifndef PT_LIGHT_H
#define PT_LIGHT_H

#include "utils/numtypes.h"

class Light {
public:
    explicit Light(const vec3 &emission) : _emission(emission) {}

    // For non-diffuse light this could depend on the incidence angle and so on...
    __device__ vec3 emission() const { return _emission; }

private:
    vec3 _emission;
};

#endif // PT_LIGHT_H
