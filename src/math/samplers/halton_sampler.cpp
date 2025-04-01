#include "halton_sampler.h"
#include "../math_utils.h"

#include <algorithm>
#include <random>
#include <spdlog/spdlog.h>

DigitPermutation::DigitPermutation(const u32 base) : base(base) {
    n_digits = 0;
    const f32 inv_base = 1.F / (f32)base;
    f32 inv_base_m = 1;
    while (1 - (base - 1) * inv_base_m < 1) {
        ++n_digits;
        inv_base_m *= inv_base;
    }

    std::random_device rd;
    // TODO: make this deterministic
    std::mt19937 g(rd());

    std::vector<u16> local_permutations;
    local_permutations.reserve(base);
    for (int i = 0; i < base; ++i) {
        local_permutations.push_back(i);
    }

    permutations = std::vector<u16>(n_digits * base);
    for (int digit_index = 0; digit_index < n_digits; ++digit_index) {
        std::ranges::shuffle(local_permutations, g);

        for (int digit_value = 0; digit_value < base; ++digit_value) {
            const auto index = digit_index * base + digit_value;
            permutations[index] = local_permutations[digit_value];
        }
    }
}

u32
DigitPermutation::permute(const u32 digit_index, const u32 digit_value) const {
    return permutations[digit_index * base + digit_value];
}

f32
HaltonSampler::radical_inverse(const u32 base_index, u64 a) {
    const u32 base = PRIMES[base_index];
    const f32 inv_base = 1.F / (f32)base;
    f32 inv_base_m = 1;
    u64 reversed_digits = 0;
    while (a) {
        const u64 next = a / base;
        const u64 digit = a - next * base;
        reversed_digits = reversed_digits * base + digit;
        inv_base_m *= inv_base;
        a = next;
    }
    return std::min(reversed_digits * inv_base_m, ONE_MINUS_EPS);
}

f32
HaltonSampler::radical_inverse_permuted(const u32 base_index, u64 a) {
    const auto &permutation = PERMUTATIONS[base_index];

    const u32 base = PRIMES[base_index];
    const f32 inv_base = 1.F / (f32)base;
    f32 inv_base_m = 1;
    u64 reversed_digits = 0;
    u32 digit_index = 0;
    while (1.F - (base - 1.F) * inv_base_m < 1.F) {
        const u64 next = a / base;
        u64 digit = a - next * base;
        digit = permutation.permute(digit_index, digit);
        reversed_digits = reversed_digits * base + digit;
        inv_base_m *= inv_base;
        a = next;
        digit_index++;
    }
    return std::min(reversed_digits * inv_base_m, ONE_MINUS_EPS);
}