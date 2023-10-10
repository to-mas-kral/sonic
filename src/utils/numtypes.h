#ifndef PT_NUMTYPES_H
#define PT_NUMTYPES_H

#include <cuda/std/cstdint>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <glm/common.hpp>
#include <glm/ext.hpp>

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

using vec2 = glm::vec2;
using vec3 = glm::vec3;
using vec4 = glm::vec4;

#endif // PT_NUMTYPES_H
