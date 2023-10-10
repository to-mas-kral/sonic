#ifndef PT_MESH_H
#define PT_MESH_H

#include "../utils/numtypes.h"
#include "../utils/shared_vector.h"

class Mesh {
public:
    Mesh(SharedVector<u32> &&a_indices, SharedVector<vec3> &&a_pos, u32 materialId,
         i32 light_id = -1)
        : indices(std::move(a_indices)), pos(std::move(a_pos)), material_id(materialId),
          light_id(light_id) {
        assert(indices.len() % 3 == 0);
    }

    Mesh(SharedVector<u32> &&a_indices, SharedVector<f32> &&a_pos, u32 a_material_id,
         i32 light_id = -1)
        : indices(std::move(a_indices)), material_id(a_material_id), light_id(light_id) {
        assert(indices.len() % 3 == 0);
        assert(a_pos.len() % 3 == 0);

        pos = SharedVector<vec3>(a_pos.len() / 3);
        for (int i = 0; i < a_pos.len(); i += 3) {
            auto p = vec3(a_pos[i], a_pos[i + 1], a_pos[i + 2]);
            pos.push(std::move(p));
        }
    }

    Mesh(Mesh const &) = delete;

    Mesh &operator=(Mesh const &) = delete;

    Mesh(Mesh &&other) noexcept {
        indices = std::move(other.indices);
        pos = std::move(other.pos);
        material_id = other.material_id;
        light_id = other.light_id;
    };

    Mesh &operator=(Mesh &&other) noexcept {
        indices = std::move(other.indices);
        pos = std::move(other.pos);
        material_id = other.material_id;
        light_id = other.light_id;

        return *this;
    };

    __host__ __device__ const SharedVector<u32> &get_indices() const { return indices; }
    __device__ const SharedVector<vec3> &get_pos() const { return pos; }
    __device__ u32 get_material_id() const { return material_id; }
    // TODO: use optional
    __device__ bool has_light() const { return light_id >= 0; }
    __device__ i32 get_light_id() const { return light_id; }

private:
    SharedVector<u32> indices;
    /// Vertices positions
    SharedVector<vec3> pos;
    u32 material_id;
    // TODO: use optional when available
    // Negative means no light
    i32 light_id;
};

#endif // PT_MESH_H
