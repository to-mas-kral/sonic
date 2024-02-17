#ifndef PT_HIT_TYPE_H
#define PT_HIT_TYPE_H

#include "../utils/basic_types.h"

enum class HitType : u32 {
    Miss = 0,
    Triangle = 1,
    Sphere = 2,
};

#endif // PT_HIT_TYPE_H
