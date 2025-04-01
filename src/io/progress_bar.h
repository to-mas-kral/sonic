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
    print(u64 current_count, u64 total_count, const std::chrono::duration<f64> elapsed) {
        const f64 progress =
            static_cast<f64>(current_count) / static_cast<f64>(total_count);

        fmt::print("\r");
        fmt::print("Sample {} / {} - {:.0f}%", current_count, total_count,
                   progress * 100);
        fmt::print("{}", bar_start);

        const u32 done_count = static_cast<u32>(static_cast<f64>(bar_length) * progress);

        for (int a = 0; a < done_count; a++) {
            fmt::print("{}", bar_filler_done);
        }

        for (int i = 0; i < bar_length - done_count; i++) {
            fmt::print("{}", bar_filler_left);
        }

        const auto duration_per_count = elapsed.count() / static_cast<f64>(current_count);

        spdlog::critical(
            "{} {} {}", static_cast<f64>(total_count - current_count), duration_per_count,
            static_cast<f64>(total_count - current_count) * duration_per_count);

        std::chrono::duration<f64> const remaining(
            static_cast<f64>(total_count - current_count) * duration_per_count);

        fmt::print(" time remaining: {:%j days and %H:%M:%S}",
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
