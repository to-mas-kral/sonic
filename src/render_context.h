#ifndef PT_RENDER_CONTEXT_H
#define PT_RENDER_CONTEXT_H

#include "camera.h"
#include "framebuffer.h"
#include "scene.h"
#include "utils/basic_types.h"
#include "scene_loader.h"

/// Render Context is a collection of data needed for the render kernels to to their
/// job.
class RenderContext {
public:
    explicit RenderContext(SceneAttribs &attribs) : attribs(attribs) {
        f32 aspect = static_cast<f32>(attribs.resx) / static_cast<f32>(attribs.resy);
        cam = Camera(attribs.fov, aspect);
        fb = Framebuffer(attribs.resx, attribs.resy);
    }

    Scene scene{};
    Camera cam;
    Framebuffer fb;
    SceneAttribs attribs;
};

#endif // PT_RENDER_CONTEXT_H
