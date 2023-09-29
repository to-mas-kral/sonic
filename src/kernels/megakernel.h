#ifndef PT_MEGAKERNEL_H
#define PT_MEGAKERNEL_H

#include "../render_context.h"

// The "megakernel" approach to path-tracing on the GPU
__global__ void render_megakernel(RenderContext *rc, u32 x, u32 y, u32 block_size) {
    u32 sample = threadIdx.x + (blockIdx.x * block_size);

    if (sample < rc->get_num_samples()) {
        if (x == 799 && y == 599) {
            printf("x: {%d}, y: {%d}, sample: {%d}\n", x, y, sample);
        }
    }
}

#endif // PT_MEGAKERNEL_H
