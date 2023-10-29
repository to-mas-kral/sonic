#ifndef PT_WAVEFRONT_RAY_H
#define PT_WAVEFRONT_RAY_H

#include "../utils/numtypes.h"
#include "ray.h"

class WavefrontRay : Ray {
public:
    __device__ WavefrontRay(const vec3 &origin, const vec3 &direction, const vec3 &o,
                            const vec3 &dir, const u32 &x, const u32 &y)
        : Ray(origin, direction), x(x), y(y) {}

    __device__ WavefrontRay(Ray ray, u32 x, u32 y) : Ray(ray), x(x), y(y) {}

    u32 x;
    u32 y;
};

#endif // PT_WAVEFRONT_RAY_H
