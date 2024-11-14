#ifndef RAYH
#define RAYH

#include "../math/transform.h"
#include "../math/vecmath.h"
#include "../utils/basic_types.h"

class Ray {
public:
    Ray(const point3 &origin, const norm_vec3 &direction)
        : m_origin(origin), m_direction(direction) {}

    void
    transform(const mat4 &transform) {
        m_origin = transform.transform_point(m_origin);
        m_direction = transform.transform_vec(m_direction).normalized();
    }

    point3
    at(const f32 t) const {
        return m_origin + (t * m_direction);
    }

    /// origin
    const point3 &
    orig() const {
        return m_origin;
    }

    /// direction
    const norm_vec3 &
    dir() const {
        return m_direction;
    }

private:
    // Origin
    point3 m_origin;
    // Direction
    norm_vec3 m_direction;
};

#endif
