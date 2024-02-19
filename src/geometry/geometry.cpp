
#include "geometry.h"

void
Geometry::add_mesh(const MeshParams &mp, Option<u32> lights_start_id) {
    u32 num_indices = mp.indices->size();
    u32 num_vertices = mp.pos->size();

    // TODO: accidentally quadratic resizing...

    u32 indices_index = meshes.indices.size();
    for (u32 &index : *mp.indices) {
        meshes.indices.push_back(index);
    }

    u32 pos_index = meshes.pos.size();
    for (auto &pos : *mp.pos) {
        meshes.pos.push_back(pos);
    }

    Option<u32> normals_index = {};
    if (mp.normals != nullptr) {
        normals_index = {meshes.normals.size()};
        assert(mp.normals->size() == mp.pos->size());

        for (auto normal : *mp.normals) {
            meshes.normals.push_back(normal);
        }
    }

    Option<u32> uvs_index = {};
    if (mp.uvs != nullptr) {
        uvs_index = {meshes.uvs.size()};
        assert(mp.uvs->size() == mp.pos->size());

        for (auto uv : *mp.uvs) {
            meshes.uvs.push_back(uv);
        }
    }

    auto mesh = Mesh(indices_index, pos_index, mp.material_id, lights_start_id,
                     num_indices, num_vertices, normals_index, uvs_index);
    meshes.meshes.push_back(mesh);
}

void
Geometry::add_sphere(SphereParams sp, Option<u32> light_id) {
    spheres.vertices.push_back(SphereVertex{
        .pos = sp.center,
        .radius = sp.radius,
    });
    spheres.material_ids.push_back(sp.material_id);
    spheres.has_light.push_back(light_id.has_value());
    if (light_id.has_value()) {
        spheres.light_ids.push_back(light_id.value());
    } else {
        spheres.light_ids.push_back(0);
    }

    spheres.num_spheres++;
}

Mesh::Mesh(u32 indices_index, u32 pos_index, u32 material_id,
           Option<u32> p_lights_start_id, u32 num_indices, u32 num_vertices,
           Option<u32> p_normals_index, Option<u32> p_uvs_index)
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
