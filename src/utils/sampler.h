#ifndef PT_SAMPLER_H
#define PT_SAMPLER_H

#include "../math/vecmath.h"
#include "basic_types.h"

// This RNG implementation was taken from Ray Tracing Gems II
u32
jenkins_hash(u32 x);

u32
init_rng(const uvec2 &pixel, const uvec2 &resolution, u32 frame);

f32
rng_uint_to_float(u32 x);

u32
xorshift(u32 &rng_state);

f32
rng(u32 &rngState);

class Sampler {
public:
    Sampler() = default;

    void
    init_frame(const uvec2 &pixel, const uvec2 &resolution, u32 p_frame, u32 spp);

    f32
    sample();

    vec2
    sample2();

    vec2
    sample_camera();

    vec3
    sample3();

private:
    u32 rand_state{0};
    u32 frame{0};
};

#endif // PT_SAMPLER_H
