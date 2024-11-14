#ifndef PT_UTILS_H
#define PT_UTILS_H

#include "../color/spectral_quantity.h"
#include "../utils/basic_types.h"

#include <algorithm>
#include <optional>

/// Randomly selects if a path should be terminated based on its throughput.
/// Roulette is only applied after the first 3 bounces.
/// Returns true if path should be terminated. If not, also returns roulette compensation.
inline std::optional<f32>
russian_roulette(const u32 depth, const f32 u, const spectral &throughput) {
    if (depth > 3) {
        const f32 survival_prob = 1.F - std::max(throughput.max_component(), 0.05F);

        if (u < survival_prob) {
            return {};
        } else {
            f32 roulette_compensation = 1.F - survival_prob;
            return roulette_compensation;
        }
    } else {
        return 1.F;
    }
}

/// Specific case where 1 sample is taken from each distribution.
inline f32
mis_power_heuristic(const f32 fpdf, const f32 gpdf) {
    return sqr(fpdf) / (sqr(fpdf) + sqr(gpdf));
}

#endif // PT_UTILS_H
