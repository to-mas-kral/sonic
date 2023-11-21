#ifndef PT_LIGHT_H
#define PT_LIGHT_H

#include "../geometry/geometry.h"
#include "../utils/basic_types.h"

struct Light {
    ShapeIndex shape;
    Emitter emitter;
};

#endif // PT_LIGHT_H
