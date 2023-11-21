#ifndef RAYH
#define RAYH

#include "../utils/basic_types.h"

class Ray {
public:
    __device__
    Ray(const vec3 &origin, const vec3 &direction)
        : o(origin), dir(direction) {}

    __device__ void
    transform(const glm::mat4 &transform) {
        o = transform * vec4(o, 1.);
        dir = transform * vec4(dir, 0.);
    }

    __device__ vec3
    at(f32 t) const {
        return o + (t * dir);
    }
    __device__ vec3
    operator()(f32 t) const {
        return (t * dir) + o;
    }

    // Origin
    vec3 o;
    // Direction
    vec3 dir;
};

#endif
