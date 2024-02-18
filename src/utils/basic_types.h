#ifndef PT_BASIC_TYPES_H
#define PT_BASIC_TYPES_H

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <tuple>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using f32 = float;
using f64 = double;

template <typename T> using Option = std::optional<T>;
template <typename T> using Span = std::span<T>;
template <typename... T> using Tuple = std::tuple<T...>;
template <typename T, u32 N> using Array = std::array<T, N>;

#endif // PT_BASIC_TYPES_H
