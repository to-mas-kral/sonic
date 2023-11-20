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
#include "utils/numtypes.h"
#include "utils/shared_vector.h"

/// Render Context is a collection of data needed for the render kernels to to their
/// job.
class RenderContext {
public:
    explicit RenderContext(SceneAttribs &attribs);

    Scene scene{};
    Camera cam;
    Framebuffer fb;
    SceneAttribs attribs;

    const u32 THREADS_DIM_SIZE = 8;
    const dim3 THREADS_DIM = dim3(THREADS_DIM_SIZE, THREADS_DIM_SIZE);
};

#endif // PT_RENDER_CONTEXT_H
