#ifndef PT_CAMERA_H
#define PT_CAMERA_H

#include "geometry/ray.h"
#include "math/vecmath.h"

class Camera {
public:
    Camera() = default;

    Camera(const f32 fov, const f32 aspect) {
        viewport_height = 2.f;
        viewport_width = viewport_height * aspect;

        const f32 axis = std::min(viewport_width, viewport_height);
        const f32 focal_length = -(axis / 2.f) / tanf((fov * (M_PIf / 180.f)) / 2.f);

        origin = point3(0.f);
        const auto horizontal = vec3(viewport_width, 0.f, 0.f);
        const auto vertical = vec3(0.f, viewport_height, 0.f);
        bottom_left =
            origin - horizontal / 2.f - vertical / 2.f - vec3(0.f, 0.f, focal_length);
    }

    Ray
    get_ray(const f32 s, const f32 t) const {
        const vec3 offset = vec3(s, t, 0.f) * vec3(viewport_width, viewport_height, 0.f);
        const point3 screencoord = bottom_left + offset;

        return Ray(origin, (screencoord - origin).normalized());
    }

    point3 origin{0.f};
    point3 bottom_left{0.f};
    f32 viewport_width{};
    f32 viewport_height{};
};

#endif
