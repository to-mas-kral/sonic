#include "scene.h"

__host__ u32
Scene::add_material(Material &&material) {
    u32 mat_id = materials.size();
    materials.push(material);
    return mat_id;
}

__host__ u32
Scene::add_texture(Texture &&texture) {
    u32 texture_id = textures.size();
    textures.push(texture);
    return texture_id;
}

__host__ void
Scene::init_light_sampler() {
    light_sampler = LightSampler(lights, geometry);
}

__host__ void
Scene::add_mesh(MeshParams mp) {
    u32 next_mesh_id = geometry.get_next_shape_index(ShapeType::Mesh);
    COption<u32> lights_start_id = {};

    if (mp.emitter.has_value()) {
        lights_start_id = COption<u32>(lights.size());

        for (u32 i = 0; i < mp.indices->size() / 3; i++) {
            auto si = ShapeIndex{
                .type = ShapeType::Mesh, .index = next_mesh_id, .triangle_index = i};

            lights.push(Light{.shape = si, .emitter = mp.emitter.value()});
        }
    }

    geometry.add_mesh(mp, lights_start_id);
}

__host__ void
Scene::add_sphere(SphereParams sp) {
    u32 next_sphere_id = geometry.get_next_shape_index(ShapeType::Sphere);
    COption<u32> light_id = {};

    if (sp.emitter.has_value()) {
        light_id = COption<u32>(lights.size());
        auto si = ShapeIndex{.type = ShapeType::Sphere, .index = next_sphere_id};

        lights.push(Light{.shape = si, .emitter = sp.emitter.value()});
    }

    geometry.add_sphere(sp, light_id);
}

Scene::~Scene() {
    for (int i = 0; i < textures.size(); i++) {
        Texture &tex = textures[i];
        tex.free();
    }
}
