#include "../render_context_common.h"

__host__ __device__ const u32 *Mesh::get_indices() const {
    return &rc->get_indices()[indices_index];
}
__host__ __device__ const vec3 *Mesh::get_pos() const {
    return &rc->get_pos()[pos_index];
}
