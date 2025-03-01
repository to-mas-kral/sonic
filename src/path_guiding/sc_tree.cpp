#include "sc_tree.h"

Reservoir &
Reservoirs::find_reservoir(const uvec2 pixel) {
    // TODO: have a bool flag for when inserting should not be done anymore so mutex
    // doesn't need to be locked
    const std::scoped_lock lock(reservoirs_mutex);

    auto min_dist = std::numeric_limits<f32>::max();
    auto min_index = 0;

    constexpr f32 MAX_DIST = 150.F;
    constexpr f32 SQUARED_MAX_DIST = sqr(MAX_DIST);

    for (u32 i = 0; i < reservoirs.size(); ++i) {
        const auto &reservoir = reservoirs[i];

        const auto res_coord = vec2(reservoir.coords.x, reservoir.coords.y);
        const auto pixel_coord = vec2(pixel.x, pixel.y);

        const auto diff = res_coord - pixel_coord;

        if (std::abs(diff.x) > MAX_DIST || std::abs(diff.y) > MAX_DIST) {
            continue;
        }

        const auto dist = diff.length_squared();

        if (dist < min_dist) {
            min_dist = dist;
            min_index = i;
        }
    }

    assert(reservoirs.capacity() == MAX_RESERVOIRS_IN_VEC);
    if ((!reservoirs.empty() && min_dist < SQUARED_MAX_DIST) ||
        reservoirs.size() >= MAX_RESERVOIRS_IN_VEC) {
        return reservoirs[min_index];
    } else {
        // Need to insert reservoir because the closest one is too far
        auto reservoir = Reservoir(u16vec2(pixel.x, pixel.y));
        reservoirs.push_back(reservoir);
        return reservoirs.back();
    }
}
