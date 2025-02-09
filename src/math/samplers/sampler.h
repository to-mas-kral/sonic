#ifndef PT_SAMPLER_H
#define PT_SAMPLER_H

#include "../../utils/basic_types.h"
#include "../../utils/hasher.h"
#include "../vecmath.h"

class Sampler {
public:
    Sampler(const uvec2 &pixel, const uvec2 &res, const u32 sample) : m_sample(sample) {
        const auto hash_base = pixel + res;
        m_halton_index_base = hash_buffer(&hash_base, 8);
    }

    f32
    sample();

    vec2
    sample2();

    vec3
    sample3();

private:
    u32 m_dimension{0};
    u32 m_sample{0};
    u64 m_halton_index_base;
};

#endif // PT_SAMPLER_H
