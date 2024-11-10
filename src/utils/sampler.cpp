#include "sampler.h"

u32
jenkins_hash(u32 x) {
    x += x << 10;
    x ^= x >> 6;
    x += x << 3;
    x ^= x >> 11;
    x += x << 15;

    return x;
}

u32
init_rng(const uvec2 &pixel, const uvec2 &resolution, const u32 frame) {
    const u32 rngState = (pixel.x + (pixel.y * resolution.x)) ^ jenkins_hash(frame);
    return jenkins_hash(rngState);
}

f32
rng_uint_to_float(const u32 x) {
    return std::bit_cast<f32>(0x3f800000 | (x >> 9)) - 1.f;
}

u32
xorshift(u32 &rng_state) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;

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
    constexpr f32 STRATUM_WIDTH = 1.f / static_cast<f32>(STRATA_SQRT_SIZE);

    const auto stratum = frame % (STRATA_SQRT_SIZE * STRATA_SQRT_SIZE);
    const auto offset_x = static_cast<f32>(stratum % STRATA_SQRT_SIZE) * STRATUM_WIDTH;
    const auto offset_y = static_cast<f32>(stratum / STRATA_SQRT_SIZE) * STRATUM_WIDTH;

    const auto in_stratum_x = rng(rand_state) / STRATUM_WIDTH;
    const auto in_stratum_y = rng(rand_state) / STRATUM_WIDTH;

    auto x = offset_x + in_stratum_x;
    auto y = offset_y + in_stratum_y;
    if (x >= 1.f) {
        x = std::nexttowardf(1.f, 0);
    }

    if (y >= 1.f) {
        y = std::nexttowardf(1.f, 0);
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
