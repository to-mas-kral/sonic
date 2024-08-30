#ifndef MORTON_H
#define MORTON_H

#include "../utils/basic_types.h"
#include "vecmath.h"

#include <tuple>

u64
morton_encode(uvec2 xy);

uvec2
morton_decode(u64 m);

#endif // MORTON_H
