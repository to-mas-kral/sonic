#ifndef PT_UTILS_H
#define PT_UTILS_H

#include "../utils/numtypes.h"

/// Randomly selects if a path should be terminated based on its throughput.
/// Roulette is only applied after the first 3 bounces.
/// Returns true if path should be terminated. If not, also returns roulette compensation.
// TODO: use cuda::std::optional when available
__device__ __forceinline__ cuda::std::tuple<bool, f32>
russian_roulette(u32 depth, curandState *rand_state, const vec3 &throughput) {
    if (depth > 3) {
        f32 u = rng(rand_state);
        f32 survival_prob =
            1.f - max(glm::max(throughput.x, throughput.y, throughput.z), 0.05f);

        if (u < survival_prob) {
            return {true, 0.f};
        } else {
            f32 roulette_compensation = 1.f - survival_prob;
            return {false, roulette_compensation};
        }
    } else {
        return {false, 1.f};
    }
}

#endif // PT_UTILS_H
