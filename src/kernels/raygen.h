#ifndef PT_RAYGEN_H
#define PT_RAYGEN_H

#include "../render_context_common.h"
#include "wavefront_common.h"

__global__ void raygen(RenderContext *rc, WavefrontState *ws);

#endif // PT_RAYGEN_H
