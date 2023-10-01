#ifndef PT_MESH_H
#define PT_MESH_H

#include "../utils/numtypes.h"
#include "../utils/shared_vector.h"

class Mesh {
public:
    Mesh(SharedVector<u32> &&a_indices, SharedVector<f32> &&pos, u32 materialId)
        : indices(std::move(a_indices)), pos(std::move(pos)), material_id(materialId) {
        assert(indices.len() % 3 == 0);
        assert(pos.len() % 3 == 0);
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

    const SharedVector<u32> &get_indices() const { return indices; }
    const SharedVector<f32> &get_pos() const { return pos; }

private:
    SharedVector<u32> indices;
    /// Vertices positions
    SharedVector<f32> pos;
    u32 material_id;
};

#endif // PT_MESH_H
