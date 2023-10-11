#ifndef PT_MEGAKERNEL_H
#define PT_MEGAKERNEL_H

#include "../render_context.h"

__global__ void render_megakernel(RenderContext *rc);

#endif // PT_MEGAKERNEL_H
