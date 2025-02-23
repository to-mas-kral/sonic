#ifndef FLOAT_COMPARISON_H
#define FLOAT_COMPARISON_H

#include "basic_types.h"

#include <cmath>

namespace sonic {
inline bool
abs_diff(const f32 a, const f32 b, const f32 threshold) {
    return std::abs(a - b) > threshold;
}
} // namespace sonic

#endif // FLOAT_COMPARISON_H
