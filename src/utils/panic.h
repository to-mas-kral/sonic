#ifndef PANIC_H
#define PANIC_H

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <source_location>
#include <type_traits>

// Taken from: https://buildingblock.ai/panic

template <class... Args> struct panic_format {
    template <class T>
    consteval panic_format( // note: consteval is what allows for compile-time
                            // checking of the
        const T &s,         //       format string
        const std::source_location loc = std::source_location::current()) noexcept
        : fmt{s}, loc{loc} {}

    fmt::format_string<Args...> fmt;
    std::source_location loc;
};

template <class... Args>
[[noreturn]] void
panic(panic_format<std::type_identity_t<Args>...>
          fmt,                 // std::type_identity_t is needed to prevent
      Args &&...args) noexcept // type deduction of the format string's
{                              // arguments.
    auto msg = fmt::format("{}:{} panic: {}\n", fmt.loc.file_name(), fmt.loc.line(),
                           fmt::format(fmt.fmt, std::forward<Args>(args)...));

    spdlog::critical(msg);
    std::abort();
}

[[noreturn]] inline void
panic(const std::source_location loc = std::source_location::current()) noexcept {
    const auto msg = fmt::format("{}:{} panic\n", loc.file_name(), loc.line());

    spdlog::critical(msg);
    std::abort();
}

#endif // PANIC_H
