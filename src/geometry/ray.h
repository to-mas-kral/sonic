#ifndef RAYH
#define RAYH

#include "../math/transform.h"
#include "../math/vecmath.h"
#include "../utils/basic_types.h"

class Ray {
public:
    Ray(const point3 &origin, const norm_vec3 &direction) : o(origin), dir(direction) {}

    void
    transform(const mat4 &transform) {
        o = transform.transform_point(o);
        dir = transform.transform_vec(dir).normalized();
    }

    point3
    at(f32 t) const {
        return o + (t * dir);
    }

    // Origin
    point3 o;
    // Direction
    norm_vec3 dir;
};

#endif
