#ifndef GEOMETRY_ALLOCATOR_H
#define GEOMETRY_ALLOCATOR_H

#include "../math/vecmath.h"

#include "../utils/hasher.h"

#include <cstdlib>
#include <set>
#include <span>
#include <spdlog/spdlog.h>
#include <type_traits>
#include <unordered_set>

/// Checks if the data type is a POD and one of the supported types
template <typename T>
concept GeometryPod =
    std::is_trivially_copyable_v<T> &&
    (std::is_same_v<T, u32> || std::is_same_v<T, vec2> || std::is_same_v<T, vec3> ||
     std::is_same_v<T, norm_vec3> || std::is_same_v<T, point3>);

template <GeometryPod T> struct GeometryBlock {
    explicit
    GeometryBlock() = default;

    explicit
    GeometryBlock(const std::span<T> &inner)
        : inner(inner) {}

    T *
    ptr() const {
        return inner.data();
    }

    friend bool
    operator==(const GeometryBlock &lhs, const GeometryBlock &rhs) {
        return (lhs.inner.size() == rhs.inner.size()) &&
               (lhs.inner.data() == rhs.inner.data() ||
                std::memcmp((void *)lhs.ptr(), (void *)rhs.ptr(),
                            lhs.inner.size_bytes()) == 0);
    }

    friend bool
    operator!=(const GeometryBlock &lhs, const GeometryBlock &rhs) {
        return !(lhs == rhs);
    }

    void
    calc_hash() {
        assert(inner.data() != nullptr);
        hash = hash_buffer(inner.data(), inner.size_bytes());
    }

    std::span<T> inner;
    XXH64_hash_t hash{0U};
};

template <GeometryPod T> struct GeometryBlockHasher {
    std::size_t
    operator()(const GeometryBlock<T> &block) const noexcept {
        return block.hash;
    }
};

// TODO: The design of this is not the best, but I don't really know how to do it better.

/// A cache for all geometry data (positions, normals, uvs, ...).
/// Attempts to detect and remove redundancies.
class GeometryStorage {
public:
    GeometryStorage() = default;

    template <GeometryPod T>
    GeometryBlock<T>
    allocate(const std::size_t count) {
        void *ptr = nullptr;
        if constexpr (std::is_same<T, point3>()) {
            // Embree requires the last element of the vertex buffer be readable by a
            // 16-byte SSE load instruction, so allocate 4 more bytes at the end.
            ptr = std::malloc(count * sizeof(T) + sizeof(f32));
            //((f32 *)ptr)[count] = 0.F;
        } else {
            ptr = std::malloc(count * sizeof(T));
        }

        assert(ptr != nullptr);
        allocated_chunks.emplace(ptr);
        return GeometryBlock<T>{std::span<T>((T *)ptr, count)};
    }

    template <GeometryPod T>
    void
    add_geom_data(GeometryBlock<T> &block) {
        block.calc_hash();

        auto insert =
            [&block,
             this](std::unordered_set<GeometryBlock<T>, GeometryBlockHasher<T>> &set) {
                const auto elem = set.find(block);
                const auto contains = elem != set.end();

                if (contains) {
                    bytes_saved += block.inner.size_bytes();
                    allocated_chunks.erase((void *)block.ptr());
                    std::free(block.ptr());
                    block.inner = elem->inner;
                    block.hash = elem->hash;
                } else {
                    const auto &ret = set.insert(block);
                    assert(ret.second);
                }
            };

        if constexpr (std::is_same_v<T, u32>) {
            insert(u32s);
        } else if constexpr (std::is_same_v<T, vec2>) {
            insert(vec2s);
        } else if constexpr (std::is_same_v<T, vec3>) {
            insert(vec3s);
        } else if constexpr (std::is_same_v<T, point3>) {
            insert(point3s);
        } else {
            static_assert(false);
        }
    }

    GeometryStorage(const GeometryStorage &other) = delete;

    GeometryStorage &
    operator=(const GeometryStorage &other) = delete;

    GeometryStorage(GeometryStorage &&other) noexcept = default;

    GeometryStorage &
    operator=(GeometryStorage &&other) noexcept = default;

    ~
    GeometryStorage() {
        spdlog::info("GeometryStorage bytes saved: {}", bytes_saved);
        for (auto *const ptr : allocated_chunks) {
            std::free(ptr);
        }
    }

private:
    u64 bytes_saved{0};
    std::set<void *> allocated_chunks;

    std::unordered_set<GeometryBlock<u32>, GeometryBlockHasher<u32>> u32s;
    std::unordered_set<GeometryBlock<vec2>, GeometryBlockHasher<vec2>> vec2s;
    std::unordered_set<GeometryBlock<vec3>, GeometryBlockHasher<vec3>> vec3s;
    // TODO: mesh normals to norm_vec3
    // std::unordered_set<GeometryBlock<norm_vec3>, GeometryBlockHasher<u3>> norm_vec3s;
    std::unordered_set<GeometryBlock<point3>, GeometryBlockHasher<point3>> point3s;
};

#endif // GEOMETRY_ALLOCATOR_H
