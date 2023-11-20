
#include "render_context.h"

#include "geometry/ray.h"

RenderContext::RenderContext(SceneAttribs &attribs) : attribs(attribs) {

    u32 blocks_x = (attribs.resx + THREADS_DIM_SIZE - 1U) / THREADS_DIM_SIZE;
    u32 blocks_y = (attribs.resy + THREADS_DIM_SIZE - 1U) / THREADS_DIM_SIZE;
    dim3 blocks_dim = dim3(blocks_x, blocks_y);

    f32 aspect = static_cast<f32>(attribs.resx) / static_cast<f32>(attribs.resy);
    cam = Camera(attribs.fov, aspect);
    fb = Framebuffer(attribs.resx, attribs.resy, blocks_dim, THREADS_DIM);
}
