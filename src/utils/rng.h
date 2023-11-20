#ifndef PT_RNG_H
#define PT_RNG_H

#include <curand_kernel.h>

__device__ inline f32 rng_curand(curandState *rand_state) {
    // curand_uniform returns (0, 1], but we need [0, 1)
    // Simply 1 - curand_uniform doesn't work !
    f32 uniform = curand_uniform(rand_state);
    // IDK if this is a good solution, but it works better than 1 - curand_uniform...
    f32 sample = nextafter(uniform, 0.f);

    assert(sample < 1.f && sample >= 0.f);

    return sample;
}

// This apparently isn't faster than curand ?!
/*// Converts unsigned integer into float int range <0; 1) by using 23 most significant
bits
// for mantissa Taken from: MARRS, Adam, Peter SHIRLEY a Ingo WALD, ed. Ray Tracing Gems
// II: Next Generation Real-Time Rendering with DXR, Vulkan, and OptiX
__device__ __forceinline__ f32 rand_u32_to_f32(u32 x) {
    return __uint_as_float(0x3f800000U | (x >> 9)) - 1.0f;
}

// Taken from:
// MARRS, Adam, Peter SHIRLEY a Ingo WALD, ed. Ray Tracing Gems II: Next Generation
// Real-Time Rendering with DXR, Vulkan, and OptiX
__device__ inline vec4 rng(u32 x, u32 y, u32 frame, u32 sample) {
    uvec4 v = uvec4(x, y, frame, sample);

    v = v * 1664525u + 1013904223u;

    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;

    v = v ^ (v >> 16u);
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;

    return vec4(rand_u32_to_f32(v.x), rand_u32_to_f32(v.y), rand_u32_to_f32(v.z),
                rand_u32_to_f32(v.w));
}*/

#endif // PT_RNG_H
