#ifndef PT_ALGS_H
#define PT_ALGS_H

#include "basic_types.h"

#include <cassert>

template <typename Accessor, typename T>
u32
binary_search_interval(size_t size, const Accessor &accessor, T val) {
    assert(size >= 2);

    size_t left = 0;
    size_t right = size - 1;

    // TODO: check this for overflows...
    while (left + 1 != right) {
        size_t half = left + (right - left) / 2;
        if (accessor(half) <= val) {
            left = half;
        } else {
            right = half;
        }
    }

    return left;
}

#endif // PT_ALGS_H
