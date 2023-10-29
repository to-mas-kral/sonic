#ifndef PT_MEGAKERNEL_H
#define PT_MEGAKERNEL_H

#include <cuda/std/atomic>

#include "../render_context_common.h"

__global__ void render_megakernel(RenderContext *rc);

#endif // PT_MEGAKERNEL_H
