#ifndef PT_BASIC_TYPES_H
#define PT_BASIC_TYPES_H

#include <cuda/std/cstdint>
#include <cuda/std/optional>
#include <cuda/std/span>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <glm/common.hpp>
#include <glm/ext.hpp>
#include <glm/gtc/type_ptr.hpp>

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

using uvec2 = glm::uvec2;
using uvec3 = glm::uvec3;
using uvec4 = glm::uvec4;

using ivec2 = glm::ivec2;
using ivec3 = glm::ivec3;
using ivec4 = glm::ivec4;

using bvec2 = glm::bvec2;
using bvec3 = glm::bvec3;
using bvec4 = glm::bvec4;

using mat3 = glm::mat3;
using mat4 = glm::mat4;

template <class T> using COption = cuda::std::optional<T>;
template <class T> using CSpan = cuda::std::span<T>;

__device__ __forceinline__ vec3
float3_to_vec(float3 f) {
    return vec3(f.x, f.y, f.z);
}
__device__ __forceinline__ float3
vec_to_float3(const vec3 &v) {
    return make_float3(v.x, v.y, v.z);
}

#endif // PT_BASIC_TYPES_H
