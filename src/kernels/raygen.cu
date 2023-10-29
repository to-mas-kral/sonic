
#include "raygen.h"

#include "../utils/rng.h"

__global__ void raygen(RenderContext *rc, WavefrontState *ws) {
    auto framebuffer = &rc->get_framebuffer();

    u32 pixel_index = framebuffer->pixel_index(blockDim, blockIdx, threadIdx);

    if (pixel_index < framebuffer->num_pixels()) {
        auto [x, y] = framebuffer->pixel_coords(blockDim, blockIdx, threadIdx);

        curandState *rand_state = &framebuffer->get_rand_state()[pixel_index];

        f32 image_x = static_cast<f32>(framebuffer->get_image_x() - 1U);
        f32 image_y = static_cast<f32>(framebuffer->get_image_y() - 1U);

        // TODO: make some sort of rng / sampler class
        f32 s_offset = rng(rand_state);
        f32 t_offset = rng(rand_state);

        f32 s = (static_cast<f32>(x) + s_offset) / image_x;
        f32 t = (static_cast<f32>(y) + t_offset) / image_y;

        const mat4 cam_to_world = rc->get_attribs().camera_to_world;

        Ray ray = rc->get_cam().get_ray(s, t);
        ray.transform_to_world(cam_to_world);

        ws->rays[pixel_index] = WavefrontRay(ray, x, y);
    }
}
