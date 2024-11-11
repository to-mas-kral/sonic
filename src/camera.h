#ifndef PT_CAMERA_H
#define PT_CAMERA_H

#include "geometry/ray.h"
#include "math/vecmath.h"

class Camera {
public:
    Camera() = default;

    Camera(const f32 fov, const f32 aspect) : viewport_height(2.F) {
        viewport_width = viewport_height * aspect;

        const f32 axis = std::min(viewport_width, viewport_height);
        const f32 focal_length = -(axis / 2.F) / tanf((fov * (M_PIf / 180.F)) / 2.F);

        origin = point3(0.F);
        const auto horizontal = vec3(viewport_width, 0.F, 0.F);
        const auto vertical = vec3(0.F, viewport_height, 0.F);
        bottom_left =
            origin - horizontal / 2.F - vertical / 2.F - vec3(0.F, 0.F, focal_length);
    }

    Ray
    get_ray(const f32 s, const f32 t) const {
        const vec3 offset = vec3(s, t, 0.F) * vec3(viewport_width, viewport_height, 0.F);
        const point3 screencoord = bottom_left + offset;

        return Ray(origin, (screencoord - origin).normalized());
    }

    point3 origin{0.F};
    point3 bottom_left{0.F};
    f32 viewport_width{};
    f32 viewport_height{};
};

#endif
