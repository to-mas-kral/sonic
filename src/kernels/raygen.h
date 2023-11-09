#ifndef PT_RAYGEN_H
#define PT_RAYGEN_H

#include "../render_context_common.h"

__device__ __forceinline__ Ray gen_ray(u32 x, u32 y, u32 res_x, u32 res_y, vec2 sample,
                                       RenderContext *rc) {
    f32 image_x = static_cast<f32>(res_x - 1U);
    f32 image_y = static_cast<f32>(res_y - 1U);

    f32 s_offset = sample.x;
    f32 t_offset = sample.y;

    f32 s = (static_cast<f32>(x) + s_offset) / image_x;
    f32 t = (static_cast<f32>(y) + t_offset) / image_y;

    const mat4 cam_to_world = rc->attribs.camera_to_world;

    Ray ray = rc->cam.get_ray(s, t);
    ray.transform(cam_to_world);

    return ray;
}

#endif // PT_RAYGEN_H
