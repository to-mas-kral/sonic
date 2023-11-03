#ifndef PT_INTERSECTION_H
#define PT_INTERSECTION_H

#include "../geometry/ray.h"
#include "../render_context_common.h"
#include "../utils/numtypes.h"

class Intersection {
public:
    /// Position
    vec3 pos;
    vec3 normal;
    /// Intersection ray parameter
    f32 t;

    Mesh *mesh;
};

__device__ __forceinline__ Ray spawn_ray(Intersection &its, const vec3 &dir) {
    // TODO: more robust floating-point error handling when spawning rays
    vec3 ray_orig = its.pos + (0.001f * its.normal);
    return Ray(ray_orig, dir);
}

#endif // PT_INTERSECTION_H
