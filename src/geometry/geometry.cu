
#include "geometry.h"

#include "../utils/basic_types.h"
#include "../utils/um_vector.h"

__host__ void
Geometry::add_mesh(const MeshParams &mp, COption<u32> lights_start_id) {
    u32 num_indices = mp.indices->size();
    u32 num_vertices = mp.pos->size();

    u32 indices_index = meshes.indices.size();
    for (int i = 0; i < mp.indices->size(); i++) {
        meshes.indices.push((*mp.indices)[i]);
    }

    u32 pos_index = meshes.pos.size();
    for (int i = 0; i < mp.pos->size(); i++) {
        meshes.pos.push((*mp.pos)[i]);
    }

    COption<u32> normals_index = {};
    if (mp.normals != nullptr) {
        normals_index = {meshes.normals.size()};
        assert(mp.normals->size() == mp.pos->size());

        for (int i = 0; i < mp.normals->size(); i++) {
            meshes.normals.push((*mp.normals)[i]);
        }
    }

    COption<u32> uvs_index = {};
    if (mp.uvs != nullptr) {
        uvs_index = {meshes.uvs.size()};
        assert(mp.uvs->size() == mp.pos->size());

        for (int i = 0; i < mp.uvs->size(); i++) {
            meshes.uvs.push((*mp.uvs)[i]);
        }
    }

    auto mesh = Mesh(indices_index, pos_index, mp.material_id, lights_start_id,
                     num_indices, num_vertices, normals_index, uvs_index);
    meshes.meshes.push(mesh);
}

__host__ void
Geometry::add_sphere(SphereParams sp, COption<u32> light_id) {
    spheres.centers.push(sp.center);
    spheres.radiuses.push(sp.radius);
    spheres.material_ids.push(sp.material_id);
    spheres.has_light.push(light_id.has_value());
    if (light_id.has_value()) {
        spheres.light_ids.push(light_id.value());
    } else {
        spheres.light_ids.push(0);
    }

    spheres.num_spheres++;
}

Mesh::Mesh(u32 indices_index, u32 pos_index, u32 material_id,
           COption<u32> p_lights_start_id, u32 num_indices, u32 num_vertices,
           COption<u32> p_normals_index, COption<u32> p_uvs_index)
    : indices_index(indices_index), pos_index(pos_index), material_id(material_id),
      num_indices(num_indices), num_vertices(num_vertices) {

    if (p_lights_start_id.has_value()) {
        lights_start_id = p_lights_start_id.value();
        has_light = true;
    }

    if (p_normals_index.has_value()) {
        normals_index = p_normals_index.value();
        has_normals = true;
    }

    if (p_uvs_index.has_value()) {
        uvs_index = p_uvs_index.value();
        has_uvs = true;
    }
}
