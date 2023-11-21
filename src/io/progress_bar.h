#ifndef PT_PROGRESS_BAR_H
#define PT_PROGRESS_BAR_H

#include <chrono>
#include <iostream>

#include <fmt/chrono.h>
#include <fmt/core.h>

#include "../utils/basic_types.h"

class ProgressBar {
public:
    // Progress is from 0..1
    void
    print(u64 current_count, u64 total_count, std::chrono::duration<f64> elapsed) {
        f64 progress = (f64)(current_count) / (f64)(total_count);

        fmt::print("\r");
        fmt::print("Sample {} / {} - {:.0f}%", current_count, total_count,
                   progress * 100);
        fmt::print("{}", bar_start);

        u32 done_count = (u32)((f64)bar_length * progress);

        for (int a = 0; a < done_count; a++) {
            fmt::print("{}", bar_filler_done);
        }

        for (int i = 0; i < bar_length - done_count; i++) {
            fmt::print("{}", bar_filler_left);
        }

        auto remaining = elapsed * (1. / progress) - elapsed;

        fmt::print(" time remaining: {:%H:%M:%S}",
                   std::chrono::floor<std::chrono::seconds>(remaining));

        fmt::print("{}", bar_end);

        std::cout << std::flush;
    }

    std::string_view bar_start = "[";
    std::string_view bar_end = "]";
    std::string_view bar_filler_done = "*";
    std::string_view bar_filler_left = " ";

private:
    u32 bar_length = 40;
};

#endif // PT_PROGRESS_BAR_H
