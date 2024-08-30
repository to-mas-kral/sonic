#include "morton.h"

#include <immintrin.h>

// See Intel's intrinsics guide for PEXT and PDEP instructions

constexpr u64 X_MASK = 0x5555555555555555U;
constexpr u64 Y_MASK = 0xAAAAAAAAAAAAAAAAU;

u64
morton_encode(const uvec2 xy) {
    u64 res = _pdep_u64(xy.x, X_MASK);
    res |= _pdep_u64(xy.y, Y_MASK);
    return res;
}

uvec2
morton_decode(const u64 m) {
    const u32 x = _pext_u64(m, X_MASK);
    const u32 y = _pext_u64(m, Y_MASK);
    return {x, y};
}
