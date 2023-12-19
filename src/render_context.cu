
#include "render_context.h"

RenderContext::RenderContext(SceneAttribs &attribs) : attribs(attribs) {
    f32 aspect = static_cast<f32>(attribs.resx) / static_cast<f32>(attribs.resy);
    cam = Camera(attribs.fov, aspect);
    fb = Framebuffer(attribs.resx, attribs.resy);
}
