#ifndef HASHER_H
#define HASHER_H

#include "basic_types.h"

#include <cstddef>
#include <type_traits>
#include <xxhash.h>

u64
inline hash_buffer(void *ptr, std::size_t size) {
    static_assert(std::is_same<u64, XXH64_hash_t>());
    return XXH64(ptr, size, 0);
}

#endif // HASHER_H
