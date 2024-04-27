#ifndef PT_RENDER_CONTEXT_H
#define PT_RENDER_CONTEXT_H

#include "camera.h"
#include "framebuffer.h"
#include "io/scene_loader.h"
#include "scene/scene.h"
#include "utils/basic_types.h"

/// Render Context is a collection of data needed for the integrators to do their job.
class RenderContext {
public:
    explicit
    RenderContext(const SceneAttribs &attribs)
        : attribs(attribs) {
        const f32 aspect = static_cast<f32>(attribs.camera_attribs.resx) /
                           static_cast<f32>(attribs.camera_attribs.resy);
        cam = Camera(attribs.camera_attribs.fov, aspect);
        fb = Framebuffer(attribs.camera_attribs.resx, attribs.camera_attribs.resy);
    }

    Scene scene{};
    Camera cam;
    Framebuffer fb;
    SceneAttribs attribs;
};

#endif // PT_RENDER_CONTEXT_H
