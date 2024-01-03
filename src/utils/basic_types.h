#ifndef PT_BASIC_TYPES_H
#define PT_BASIC_TYPES_H

#include <cuda/std/cstdint>
#include <cuda/std/optional>
#include <cuda/std/span>

using u8 = cuda::std::uint8_t;
using u16 = cuda::std::uint16_t;
using u32 = cuda::std::uint32_t;
using u64 = cuda::std::uint64_t;

using i8 = cuda::std::int8_t;
using i16 = cuda::std::int16_t;
using i32 = cuda::std::int32_t;
using i64 = cuda::std::int64_t;

using f32 = float;
using f64 = double;

template <typename T> using COption = cuda::std::optional<T>;
template <typename T> using CSpan = cuda::std::span<T>;
template <typename... T> using CTuple = cuda::std::tuple<T...>;
template <typename T, u32 N> using CArray = cuda::std::array<T, N>;
constexpr cuda::std::nullopt_t none = cuda::std::nullopt;

#endif // PT_BASIC_TYPES_H
