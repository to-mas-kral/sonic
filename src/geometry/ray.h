#ifndef RAYH
#define RAYH

#include "../utils/numtypes.h"

class Ray {
public:
    __device__ Ray(const vec3 &origin, const vec3 &direction) {
        o = origin;
        dir = direction;
    }

    __device__ void transform_to_world(const glm::mat4 &cam_to_world) {
        o = cam_to_world * vec4(o, 1.);
        dir = cam_to_world * vec4(dir, 0.);
    }

    __device__ vec3 at(f32 t) const { return o + (t * dir); }
    __device__ vec3 operator()(f32 t) const { return (t * dir) + o; }

    // Origin
    vec3 o;
    // Direction
    vec3 dir;
};

#endif
