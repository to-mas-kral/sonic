#ifndef PT_RAYGEN_H
#define PT_RAYGEN_H

#include "../render_context.h"
#include "../math/vecmath.h"

__device__ __forceinline__ Ray
gen_ray(u32 x, u32 y, u32 res_x, u32 res_y, const vec2 &sample, const Camera &cam,
        const mat4 &cam_to_world) {
    f32 image_x = static_cast<f32>(res_x - 1U);
    f32 image_y = static_cast<f32>(res_y - 1U);

    f32 s_offset = sample.x;
    f32 t_offset = sample.y;

    f32 s = (static_cast<f32>(x) + s_offset) / image_x;
    f32 t = (static_cast<f32>(y) + t_offset) / image_y;

    Ray ray = cam.get_ray(s, t);
    ray.transform(cam_to_world);

    return ray;
}

#endif // PT_RAYGEN_H
