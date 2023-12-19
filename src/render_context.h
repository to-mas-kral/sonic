#ifndef PT_RENDER_CONTEXT_H
#define PT_RENDER_CONTEXT_H

#include <cuda/std/atomic>
#include <cuda/std/optional>

#include "camera.h"
#include "emitter.h"
#include "envmap.h"
#include "framebuffer.h"
#include "geometry/geometry.h"
#include "geometry/ray.h"
#include "material.h"
#include "scene.h"
#include "scene_loader.h"
#include "utils/basic_types.h"
#include "utils/um_vector.h"

/// Render Context is a collection of data needed for the render kernels to to their
/// job.
class RenderContext {
public:
    explicit RenderContext(SceneAttribs &attribs);

    Scene scene{};
    Camera cam;
    Framebuffer fb;
    SceneAttribs attribs;
};

#endif // PT_RENDER_CONTEXT_H
