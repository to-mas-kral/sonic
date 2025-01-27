#include "thread_pool.h"

#include <xmmintrin.h>

static constexpr u32 TILE_SIZE = 8;

struct Tile {
    static Tile
    make_from_tile_index(u32 tile_index, uvec2 dimensions);

    u32 start_x;
    u32 end_x;
    u32 start_y;
    u32 end_y;
};

Tile
Tile::make_from_tile_index(const u32 tile_index, const uvec2 dimensions) {
    const u32 tiles_per_row = dimensions.x / TILE_SIZE;
    const u32 tile_on_column = tile_index % tiles_per_row;
    const u32 tile_on_row = tile_index / tiles_per_row;

    const u32 start_x = tile_on_column * TILE_SIZE;
    const u32 start_y = tile_on_row * TILE_SIZE;

    u32 end_x = start_x + TILE_SIZE - 1;
    u32 end_y = start_y + TILE_SIZE - 1;

    if (end_x >= dimensions.x) {
        end_x = dimensions.x - 1;
    }

    if (end_y >= dimensions.y) {
        end_y = dimensions.y - 1;
    }

    return Tile{
        .start_x = start_x,
        .end_x = end_x,
        .start_y = start_y,
        .end_y = end_y,
    };
}

namespace {
u32
get_num_threads(const Settings &settings) {
    if (settings.num_threads != 0) {
        return settings.num_threads;
    }

    u32 num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        spdlog::warn("Could not get the number of CPU cores, defaulting to 1 thread");
        num_threads = 1;
    }

    return num_threads;
}
} // namespace

ThreadPool::
ThreadPool(const SceneAttribs &scene_attribs, Integrator *integrator,
              const Settings &settings)
    : integrator{integrator}, num_threads(get_num_threads(settings)),
      start_work{num_threads + 1}, end_work{num_threads + 1},
      dimensions(uvec2(scene_attribs.film.resx, scene_attribs.film.resy)) {
    threads.reserve(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        auto t = std::jthread([i, this] { render(i); });
        threads.push_back(std::move(t));
    }

    const u64 frame_area = scene_attribs.film.resx * scene_attribs.film.resy;
    constexpr u64 tile_area = TILE_SIZE * TILE_SIZE;

    tiles_per_frame = (frame_area + tile_area - 1) / tile_area;
    tile_counter = tiles_per_frame;
}

void
ThreadPool::stop() {
    should_stop = true;
    start_work.arrive_and_wait();
}

void
ThreadPool::start_new_frame() {
    tile_counter = 0;

    start_work.arrive_and_wait();
    end_work.arrive_and_wait();
}

void
ThreadPool::render(u32 thread_id) {
    // Recommended by the Embree manual
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    while (true) {
        start_work.arrive_and_wait();

        if (should_stop) {
            return;
        }

        while (true) {
            const u32 tile_index = tile_counter.fetch_add(1, std::memory_order::relaxed);

            if (tile_index < tiles_per_frame) {
                const auto tile = Tile::make_from_tile_index(tile_index, dimensions);

                for (u32 x = tile.start_x; x <= tile.end_x; ++x) {
                    for (u32 y = tile.start_y; y <= tile.end_y; ++y) {
                        integrator->integrate_pixel(uvec2(x, y));
                    }
                }
            } else {
                break;
            }
        }

        end_work.arrive_and_wait();
    }
}
