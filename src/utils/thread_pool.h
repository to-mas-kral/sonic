#ifndef PT_RENDER_THREADS_H
#define PT_RENDER_THREADS_H

#include "../integrator/integrator.h"
#include "../scene/scene_attribs.h"
#include "basic_types.h"

#include <atomic>
#include <barrier>
#include <thread>
#include <vector>

class ThreadPool {
public:
    ThreadPool(const SceneAttribs &scene_attribs, Integrator *integrator,
                  const Settings &settings);

    void
    stop();

    void
    start_new_frame();

    void
    render(u32 thread_id);

private:
    Integrator *integrator;

    u32 num_threads;
    std::vector<std::jthread> threads;
    bool should_stop = false;

    std::barrier<> start_work;
    std::barrier<> end_work;

    /// 8x8 tiles
    uvec2 dimensions;
    u32 tiles_per_frame;
    std::atomic<u32> tile_counter;
};

#endif // PT_RENDER_THREADS_H
