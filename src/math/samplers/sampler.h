#ifndef PT_SAMPLER_H
#define PT_SAMPLER_H

#include "../../utils/basic_types.h"
#include "../../utils/hasher.h"
#include "../vecmath.h"
#include "halton_sampler.h"

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

/// Creates multiple samples in a row from the same dimension (or dimensions).
class DimensionSampler {
public:
    explicit
    DimensionSampler(const u64 seed = 0)
        : seed(seed) {}

    f32
    sample() {
        const auto x = HaltonSampler::radical_inverse_permuted(0, seed + m_sample);
        m_sample++;
        return x;
    }

    vec2
    sample2() {
        const auto x = HaltonSampler::radical_inverse_permuted(0, seed + m_sample);
        const auto y = HaltonSampler::radical_inverse_permuted(1, seed + m_sample);
        m_sample++;
        return {x, y};
    }

    vec3
    sample3() {
        const auto x = HaltonSampler::radical_inverse_permuted(0, seed + m_sample);
        const auto y = HaltonSampler::radical_inverse_permuted(1, seed + m_sample);
        const auto z = HaltonSampler::radical_inverse_permuted(2, seed + m_sample);
        m_sample++;
        return {x, y, z};
    }

private:
    u64 seed{0};
    u64 m_sample{0};
};

#endif // PT_SAMPLER_H
