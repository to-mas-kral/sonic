#ifndef PT_RENDER_THREADS_H
#define PT_RENDER_THREADS_H

#include "../integrator/integrator.h"
#include "../io/scene_loader.h"
#include "basic_types.h"

#include <atomic>
#include <thread>
#include <vector>

class RenderThreads {
public:
    RenderThreads(const SceneAttribs &scene_attribs, Integrator *integrator);

    void
    schedule_stop();

    void
    start_new_frame();

    void
    spinlock();

    void
    render();

private:
    Integrator *integrator;

    u32 num_threads;
    std::vector<std::jthread> threads{};
    bool should_stop = false;

    std::atomic<u32> threads_done_in_frame = 0;

    /// 4x4 tiles
    uvec2 dimensions;
    u32 tiles_per_frame;
    std::atomic<u32> tile_counter;
};

#endif // PT_RENDER_THREADS_H
