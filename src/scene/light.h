#ifndef PT_LIGHT_H
#define PT_LIGHT_H

#include "../geometry/geometry.h"
#include "../utils/basic_types.h"

struct Light {
    ShapeIndex shape;
    Emitter emitter;

    f32
    power(const Geometry &geom) const {
        // FIXME: account for two-sided...
        f32 area = geom.shape_area(shape);
        return M_PI * emitter.power() * area;
    }
};

#endif // PT_LIGHT_H
