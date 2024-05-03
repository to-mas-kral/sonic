#ifndef TEXTURE_ID_H
#define TEXTURE_ID_H

#include "../utils/basic_types.h"

struct TextureId {
    explicit
    TextureId(const u32 inner)
        : inner(inner) {}

    u32 inner;
};

#endif // TEXTURE_ID_H
