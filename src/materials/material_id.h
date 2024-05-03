#ifndef MATERIAL_ID_H
#define MATERIAL_ID_H

#include "../utils/basic_types.h"

struct MaterialId {
    explicit
    MaterialId(const u32 id)
        : inner(id) {}

    u32 inner;
};

#endif // MATERIAL_ID_H
