#include "scene.h"

#include "../color/spectral_data.h"

Scene::
Scene() {
    builtin_spectra.insert({"metal-Au-eta", Spectrum(AU_ETA, &spectrum_allocator)});
    builtin_spectra.insert({"metal-Au-k", Spectrum(AU_K, &spectrum_allocator)});
    builtin_spectra.insert({"metal-Ag-eta", Spectrum(AG_ETA, &spectrum_allocator)});
    builtin_spectra.insert({"metal-Ag-k", Spectrum(AG_K, &spectrum_allocator)});
    builtin_spectra.insert({"metal-Cu-eta", Spectrum(CU_ETA, &spectrum_allocator)});
    builtin_spectra.insert({"metal-Cu-k", Spectrum(CU_K, &spectrum_allocator)});

    builtin_spectra.insert({"glass-BK7", Spectrum(GLASS_BK7_ETA, &spectrum_allocator)});

    for (const auto &spectrum : builtin_spectra) {
        builtin_textures.insert(
            {spectrum.first,
             add_texture(Texture::make_constant_texture(spectrum.second))});
    }

    builtin_textures.insert(
        {"reflectance", add_texture(Texture::make_constant_texture(0.5f))});
    builtin_textures.insert(
        {"roughness", add_texture(Texture::make_constant_texture(0.f))});
    builtin_textures.insert(
        {"eta-dielectric", add_texture(Texture::make_constant_texture(1.5f))});
    builtin_textures.insert({"eta-conductor", add_texture(Texture::make_constant_texture(
                                                  builtin_spectra.at("metal-Cu-eta")))});
    builtin_textures.insert({"k-conductor", add_texture(Texture::make_constant_texture(
                                                builtin_spectra.at("metal-Cu-k")))});
}

MaterialId
Scene::add_material(const Material &material) {
    const u32 mat_id = materials.size();
    materials.push_back(material);
    return MaterialId(mat_id);
}

TextureId
Scene::add_texture(const Texture &texture) {
    const u32 texture_id = textures.size();
    textures.push_back(texture);
    return TextureId{texture_id};
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

Scene::~
Scene() {
    for (auto &tex : textures) {
        tex.free();
    }
}
