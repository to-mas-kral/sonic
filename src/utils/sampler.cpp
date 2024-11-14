#include "sampler.h"

u32
jenkins_hash(u32 x) {
    x += x << 10U;
    x ^= x >> 6U;
    x += x << 3U;
    x ^= x >> 11U;
    x += x << 15U;

    return x;
}

u32
init_rng(const uvec2 &pixel, const uvec2 &resolution, const u32 frame) {
    const u32 rngState = (pixel.x + (pixel.y * resolution.x)) ^ jenkins_hash(frame);
    return jenkins_hash(rngState);
}

f32
rng_uint_to_float(const u32 x) {
    return std::bit_cast<f32>(0x3f800000U | (x >> 9U)) - 1.F;
}

u32
xorshift(u32 &rng_state) {
    rng_state ^= rng_state << 13U;
    rng_state ^= rng_state >> 17U;
    rng_state ^= rng_state << 5U;

    return rng_state;
}

f32
rng(u32 &rngState) {
    return rng_uint_to_float(xorshift(rngState));
}

void
Sampler::init_frame(const uvec2 &pixel, const uvec2 &resolution, const u32 p_frame,
                    const u32 spp) {
    rand_state = init_rng(pixel, resolution, p_frame);
    frame = p_frame;
}

f32
Sampler::sample() {
    return rng(rand_state);
}

vec2
Sampler::sample2() {
    // The order has to be right...
    const auto r1 = rng(rand_state);
    const auto r2 = rng(rand_state);
    return vec2(r1, r2);
}

vec2
Sampler::sample_camera() {
    constexpr i32 STRATA_SQRT_SIZE = 8;
    constexpr f32 STRATUM_WIDTH = 1.F / static_cast<f32>(STRATA_SQRT_SIZE);

    const auto stratum = frame % (STRATA_SQRT_SIZE * STRATA_SQRT_SIZE);
    const auto offset_x = static_cast<f32>(stratum % STRATA_SQRT_SIZE) * STRATUM_WIDTH;
    const auto offset_y = static_cast<f32>(stratum / STRATA_SQRT_SIZE) * STRATUM_WIDTH;

    const auto in_stratum_x = rng(rand_state) / STRATUM_WIDTH;
    const auto in_stratum_y = rng(rand_state) / STRATUM_WIDTH;

    auto x = offset_x + in_stratum_x;
    auto y = offset_y + in_stratum_y;
    if (x >= 1.F) {
        x = std::nexttowardf(1.F, 0);
    }

    if (y >= 1.F) {
        y = std::nexttowardf(1.F, 0);
    }

    return vec2(x, y);
}

vec3
Sampler::sample3() {
    const auto r1 = rng(rand_state);
    const auto r2 = rng(rand_state);
    const auto r3 = rng(rand_state);
    return vec3(r1, r2, r3);
}
