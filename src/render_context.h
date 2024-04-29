#ifndef PT_RENDER_CONTEXT_H
#define PT_RENDER_CONTEXT_H

#include "camera.h"
#include "framebuffer.h"
#include "scene/scene.h"
#include "scene/scene_attribs.h"
#include "utils/basic_types.h"

/// Render Context is a collection of data needed for the integrators to do their job.
class RenderContext {
public:
    explicit
    RenderContext(Scene &&scene)
        : attribs(scene.attribs), scene{std::move(scene)} {
        const f32 aspect =
            static_cast<f32>(attribs.film.resx) / static_cast<f32>(attribs.film.resy);
        cam = Camera(attribs.camera.fov, aspect);
        fb = Framebuffer(attribs.film.resx, attribs.film.resy);
    }

    SceneAttribs attribs;
    Scene scene{};
    Camera cam;
    Framebuffer fb;
};

#endif // PT_RENDER_CONTEXT_H
