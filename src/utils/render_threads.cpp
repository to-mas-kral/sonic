#include "render_threads.h"

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
Tile::make_from_tile_index(u32 tile_index, uvec2 dimensions) {
    u32 tiles_per_row = dimensions.x / TILE_SIZE;
    u32 tile_on_column = tile_index % tiles_per_row;
    u32 tile_on_row = tile_index / tiles_per_row;

    u32 start_x = tile_on_column * TILE_SIZE;
    u32 start_y = tile_on_row * TILE_SIZE;

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

RenderThreads::RenderThreads(const SceneAttribs &scene_attribs, Integrator *integrator)
    : integrator{integrator}, num_threads(std::thread::hardware_concurrency()),
      dimensions(uvec2(scene_attribs.resx, scene_attribs.resy)) {
    if (num_threads == 0) {
        num_threads = 4;
    }

    threads.reserve(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        auto t = std::jthread(&RenderThreads::render, this);
        threads.push_back(std::move(t));
    }

    u64 frame_area = scene_attribs.resx * scene_attribs.resy;
    u64 tile_area = TILE_SIZE * TILE_SIZE;

    tiles_per_frame = (frame_area + tile_area - 1) / tile_area;
    tile_counter = tiles_per_frame;
}

void
RenderThreads::schedule_stop() {
    should_stop = true;
    threads_done_in_frame = 0;
}

void
RenderThreads::start_new_frame() {
    tile_counter = 0;
    threads_done_in_frame = 0;

    while (threads_done_in_frame != num_threads) {
        // TODO: handle this synchronization better
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(100ms);
    }
}

void
RenderThreads::spinlock() {}

void
RenderThreads::render() {
    while (!should_stop) {
        const u32 tile_index = tile_counter.fetch_add(1);

        if (tile_index < tiles_per_frame) {
            auto tile = Tile::make_from_tile_index(tile_index, dimensions);

            for (int x = tile.start_x; x <= tile.end_x; ++x) {
                for (int y = tile.start_y; y <= tile.end_y; ++y) {
                    integrator->integrate_pixel(uvec2(x, y));
                }
            }
        } else {
            threads_done_in_frame.fetch_add(1);

            // FIXME: not 100% sure, but I think this requires a sync point to be actually
            // correct
            //  or maybe add another spinlock at the start of the frame...
            while (threads_done_in_frame != 0) {
                spinlock();
            }
        }
    }
}
