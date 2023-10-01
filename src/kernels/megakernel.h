#ifndef PT_MEGAKERNEL_H
#define PT_MEGAKERNEL_H

#include "../render_context.h"

__device__ f32 rng(curandState *rand_state) { return curand_uniform(rand_state); }

/// The "megakernel" approach to path-tracing on the GPU
__global__ void render_megakernel(RenderContext *rc, u32 x, u32 y) {
    u32 sample = threadIdx.x + (blockIdx.x * blockDim.x);

    if (sample < rc->get_num_samples()) {
        auto pixel_index = rc->pixel_index(x, y);

        // TODO: might be better to just copy come PCG implementation, this uses a LOT of
        // space and init time..
        curandState rand_state;
        curand_init(1984 + pixel_index * sample, 0, 0, &rand_state);

        auto color = vec3(rng(&rand_state), rng(&rand_state), rng(&rand_state));
        rc->get_sample_accum()[sample] = color;
    }
}

#endif // PT_MEGAKERNEL_H
