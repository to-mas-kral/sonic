#ifndef PT_RAYGEN_H
#define PT_RAYGEN_H

#include "../render_context_common.h"

__device__ __forceinline__ Ray gen_ray(u32 x, u32 y, Framebuffer *fb, RenderContext *rc) {
    auto pixel_index = fb->pixel_index(x, y);
    curandState *rand_state = &fb->get_rand_state()[pixel_index];

    f32 image_x = static_cast<f32>(fb->get_image_x() - 1U);
    f32 image_y = static_cast<f32>(fb->get_image_y() - 1U);

    // TODO: make some sort of rng / sampler class
    f32 s_offset = rng(rand_state);
    f32 t_offset = rng(rand_state);

    f32 s = (static_cast<f32>(x) + s_offset) / image_x;
    f32 t = (static_cast<f32>(y) + t_offset) / image_y;

    const mat4 cam_to_world = rc->get_attribs().camera_to_world;

    Ray ray = rc->get_cam().get_ray(s, t);
    ray.transform_to_world(cam_to_world);

    return ray;
}

#endif // PT_RAYGEN_H
