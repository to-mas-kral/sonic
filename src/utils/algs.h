#ifndef PT_ALGS_H
#define PT_ALGS_H

#include "basic_types.h"

#include <cassert>

// TODO: replace with std::<something>

template <typename Accessor, typename T>
u32
binary_search_interval(const std::size_t size, const Accessor &accessor, T val) {
    assert(size >= 2);

    std::size_t left = 0;
    std::size_t right = size - 1;

    // TODO: check this for overflows...
    while (left + 1 != right) {
        std::size_t half = left + (right - left) / 2;
        const auto accessed = accessor(half); 
        if (accessed <= val) {
            left = half;
        } else {
            right = half;
        }
    }

    return left;
}

#endif // PT_ALGS_H
