#ifndef PT_SAMPLER_H
#define PT_SAMPLER_H

#include "basic_types.h"
#include "../math/vecmath.h"

// This RNG implementation was taken from Ray Tracing Gems II
__device__ __forceinline__ u32
jenkins_hash(u32 x) {
    x += x << 10;
    x ^= x >> 6;
    x += x << 3;
    x ^= x >> 11;
    x += x << 15;

    return x;
}

__device__ __forceinline__ u32
init_rng(const uvec2 &pixel, const uvec2 &resolution, u32 frame) {
    u32 rngState = (pixel.x + (pixel.y * resolution.x)) ^ jenkins_hash(frame);
    return jenkins_hash(rngState);
}

__device__ __forceinline__ f32
rng_uint_to_float(u32 x) {
    return __uint_as_float(0x3f800000 | (x >> 9)) - 1.f;
}

__device__ __forceinline__ u32
xorshift(u32 &rng_state) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;

    return rng_state;
}

__device__ __forceinline__ f32
rng(u32 &rngState) {
    return rng_uint_to_float(xorshift(rngState));
}

class Sampler {
public:
    Sampler() = default;

    __device__ __forceinline__ f32
    init_frame(const uvec2 &pixel, const uvec2 &resolution, u32 frame) {
        rand_state = init_rng(pixel, resolution, frame);
    }

    __device__ __forceinline__ f32
    sample() {
        return rng(rand_state);
    }

    __device__ __forceinline__ vec2
    sample2() {
        // The order has to be right...
        auto r1 = rng(rand_state);
        auto r2 = rng(rand_state);
        return vec2(r1, r2);
    }

    __device__ __forceinline__ vec3
    sample3() {
        auto r1 = rng(rand_state);
        auto r2 = rng(rand_state);
        auto r3 = rng(rand_state);
        return vec3(r1, r2, r3);
    }

private:
    u32 rand_state{0};
};

#endif // PT_SAMPLER_H
