#ifndef PT_RNG_H
#define PT_RNG_H

#include <curand_kernel.h>

__device__ inline f32 rng(curandState *rand_state) {
    // curand_uniform exlcudes 0 but includes 1, so invert.
    return 1.f - curand_uniform(rand_state);
}

#endif // PT_RNG_H
