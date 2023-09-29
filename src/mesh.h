#ifndef PT_MESH_H
#define PT_MESH_H

#include "numtypes.h"
#include "utils/shared_vector.h"

class Mesh {
public:
    Mesh(SharedVector<u32> &&a_indices, SharedVector<f32> &&pos, u32 materialId)
            : indices(std::move(a_indices)),
              pos(std::move(pos)),
              material_id(materialId) {
    }

    Mesh(Mesh const &) = delete;

    Mesh &operator=(Mesh const &) = delete;

    Mesh(Mesh &&other) noexcept {
        indices = std::move(other.indices);
        pos = std::move(other.pos);
        material_id = other.material_id;
    };

    Mesh &operator=(Mesh &&other) noexcept {
        indices = std::move(other.indices);
        pos = std::move(other.pos);
        material_id = other.material_id;

        return *this;
    };

private:
    SharedVector<u32> indices{};
    /// Vertices positions
    SharedVector<f32> pos{};
    u32 material_id{0};
};

#endif //PT_MESH_H
