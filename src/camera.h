#ifndef CAMERAH
#define CAMERAH

#include "geometry/ray.h"

class Camera {
public:
    Camera() {}

    __host__ Camera(float fov, float aspect) {
        viewport_height = 2.f;
        viewport_width = viewport_height * aspect;

        // From PBRT docs about FOV:
        // This is the spread angle of the viewing frustum along the narrower of the
        // image's width and height.
        f32 narrower = std::min(viewport_width, viewport_height);
        f32 focal_length = -(narrower / 2.f) / tanf((fov * (M_PIf / 180.f)) / 2.f);

        origin = vec3(0.f, 0.f, 0.f);
        vec3 horizontal = vec3(viewport_width, 0.f, 0.f);
        vec3 vertical = vec3(0.f, viewport_height, 0.f);
        bottom_left =
            origin - horizontal / 2.f - vertical / 2.f - vec3(0.f, 0.f, focal_length);
    }

    __device__ Ray get_ray(float s, float t) const {
        vec3 offset = vec3(s, t, 0.f) * vec3(viewport_width, viewport_height, 0.f);
        vec3 screencoord = bottom_left + offset;

        return Ray(origin, glm::normalize(screencoord - origin));
    }

    vec3 origin;
    vec3 bottom_left;
    f32 viewport_width;
    f32 viewport_height;
};

#endif
