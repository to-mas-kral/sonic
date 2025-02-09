#include "sampler.h"

#include "halton_sampler.h"

f32
Sampler::sample() {
    if (m_dimension > MAX_HALTON_DIM) {
        m_dimension = 2;
    }

    const auto x = HaltonSampler::radical_inverse_permuted(
        m_dimension, m_halton_index_base + m_sample);
    m_dimension++;

    return x;
}

vec2
Sampler::sample2() {
    if (m_dimension + 1 > MAX_HALTON_DIM) {
        m_dimension = 2;
    }

    const auto x = HaltonSampler::radical_inverse_permuted(
        m_dimension, m_halton_index_base + m_sample);
    const auto y = HaltonSampler::radical_inverse_permuted(
        m_dimension + 1, m_halton_index_base + m_sample);
    m_dimension += 2;

    return {x, y};
}

vec3
Sampler::sample3() {
    if (m_dimension + 2 > MAX_HALTON_DIM) {
        m_dimension = 2;
    }

    const auto x = HaltonSampler::radical_inverse_permuted(
        m_dimension, m_halton_index_base + m_sample);
    const auto y = HaltonSampler::radical_inverse_permuted(
        m_dimension + 1, m_halton_index_base + m_sample);
    const auto z = HaltonSampler::radical_inverse_permuted(
        m_dimension + 2, m_halton_index_base + m_sample);
    m_dimension += 3;

    return {x, y, z};
}
