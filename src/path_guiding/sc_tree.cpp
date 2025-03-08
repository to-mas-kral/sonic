#include "sc_tree.h"

const Reservoir *
Reservoirs::find_reservoir_inner(const uvec2 pixel, bool is_training_phase) {
    auto min_dist = std::numeric_limits<f32>::max();
    i32 min_index = -1;

    constexpr f32 MAX_DIST = 150;
    constexpr f32 SQUARED_MAX_DIST = sqr(MAX_DIST);

    for (i32 i = 0; i < num_reservoirs; ++i) {
        const auto res_coord = vec2(x_coords[i], y_coords[i]);
        const auto pixel_coord = vec2(pixel.x, pixel.y);

        const auto diff = res_coord - pixel_coord;
        const auto dist = diff.length_squared();

        if (dist < SQUARED_MAX_DIST && dist < min_dist) {
            min_dist = dist;
            min_index = i;
        }
    }

    if (!is_training_phase && min_index == -1) {
        return nullptr;
    }

    if (min_index == -1) {
        if (num_reservoirs == MAX_RESERVOIRS_PER_MAT - 1) {
            spdlog::critical("Max number of reservoirs exceeded");
            return &reservoirs[0];
        }

        x_coords[num_reservoirs] = pixel.x;
        y_coords[num_reservoirs] = pixel.y;
        reservoirs[num_reservoirs] = Reservoir();
        num_reservoirs++;
        return &reservoirs[num_reservoirs - 1];
    } else {
        return &reservoirs[min_index];
    }
}
const Reservoir *
Reservoirs::find_reservoir(const uvec2 pixel, const bool is_training_phase) {
    if (is_training_phase) {
        const std::scoped_lock lock(reservoirs_mutex);
        return find_reservoir_inner(pixel, is_training_phase);
    } else {
        return find_reservoir_inner(pixel, is_training_phase);
    }
}
