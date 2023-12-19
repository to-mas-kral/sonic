#ifndef PT_SAMPLER_H
#define PT_SAMPLER_H

#include "basic_types.h"
#include "rng.h"

class Sampler {
public:
    Sampler() = default;

    explicit Sampler(curandState rand_state) : rand_state(rand_state) {}

    __device__ __forceinline__ f32
    sample() {
        return rng_curand(&rand_state);
    }

    __device__ __forceinline__ vec2
    sample2() {
        return vec2(rng_curand(&rand_state), rng_curand(&rand_state));
    }

    __device__ __forceinline__ vec3
    sample3() {
        return vec3(rng_curand(&rand_state), rng_curand(&rand_state),
                    rng_curand(&rand_state));
    }

    __device__ __forceinline__ curandState *
    get_rand_state() {
        return &rand_state;
    }

private:
    curandState rand_state;
};

#endif // PT_SAMPLER_H
