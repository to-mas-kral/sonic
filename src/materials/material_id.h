#ifndef MATERIAL_ID_H
#define MATERIAL_ID_H

#include <unordered_map>

#include "../utils/basic_types.h"

struct MaterialId {
    explicit
    MaterialId(const u32 id)
        : inner(id) {}

    bool
    operator==(const MaterialId &) const = default;

    u32 inner;
};

template <> struct std::hash<MaterialId> {
    std::size_t
    operator()(const MaterialId &id) const noexcept {
        return id.inner;
    }
};

#endif // MATERIAL_ID_H
