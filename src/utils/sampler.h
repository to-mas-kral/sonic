#ifndef PT_SAMPLER_H
#define PT_SAMPLER_H

#include "numtypes.h"
#include "rng.h"

class Sampler {
public:
    Sampler() = default;

    explicit Sampler(curandState rand_state) : rand_state(rand_state) {}

    __device__ __forceinline__ f32 sample() { return rng_curand(&rand_state); }
    __device__ __forceinline__ curandState *get_rand_state() { return &rand_state; }

private:
    curandState rand_state;
};

#endif // PT_SAMPLER_H
