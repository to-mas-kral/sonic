#include "scene.h"

u32
Scene::add_material(Material &&material) {
    u32 mat_id = materials.size();
    materials.push_back(material);
    return mat_id;
}

u32
Scene::add_texture(Texture &&texture) {
    u32 texture_id = textures.size();
    textures.push_back(texture);
    return texture_id;
}

void
Scene::init_light_sampler() {
    light_sampler = LightSampler(lights, geometry);
}

void
Scene::add_mesh(MeshParams mp) {
    u32 next_mesh_id = geometry.get_next_shape_index(ShapeType::Mesh);
    Option<u32> lights_start_id = {};

    if (mp.emitter.has_value()) {
        lights_start_id = Option<u32>(lights.size());

        for (u32 i = 0; i < mp.indices->size() / 3; i++) {
            auto si = ShapeIndex{
                .type = ShapeType::Mesh, .index = next_mesh_id, .triangle_index = i};

            lights.push_back(Light{.shape = si, .emitter = mp.emitter.value()});
        }
    }

    geometry.add_mesh(mp, lights_start_id);
}

void
Scene::add_sphere(SphereParams sp) {
    u32 next_sphere_id = geometry.get_next_shape_index(ShapeType::Sphere);
    Option<u32> light_id = {};

    if (sp.emitter.has_value()) {
        light_id = Option<u32>(lights.size());
        auto si = ShapeIndex{.type = ShapeType::Sphere, .index = next_sphere_id};

        lights.push_back(Light{.shape = si, .emitter = sp.emitter.value()});
    }

    geometry.add_sphere(sp, light_id);
}

Scene::~Scene() {
    for (auto &tex : textures) {
        tex.free();
    }
}
