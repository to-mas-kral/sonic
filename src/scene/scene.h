#ifndef PT_SCENE_H
#define PT_SCENE_H

#include "../geometry/geometry.h"
#include "../integrator/light_sampler.h"
#include "../materials/material.h"
#include "envmap.h"
#include "light.h"
#include "scene_attribs.h"
#include "texture.h"
#include "texture_id.h"

struct Scene {
    Scene();

    ~
    Scene();

    void
    set_envmap(Envmap &&a_envmap) {
        envmap = std::move(a_envmap);
    }

    void
    add_mesh(MeshParams mp);
    void
    add_sphere(SphereParams sp);

    MaterialId
    add_material(const Material &material);

    TextureId
    add_texture(const Texture &texture);

    void
    init_light_sampler();

    Option<LightSample>
    sample_lights(f32 sample) {
        return light_sampler.sample(lights, sample);
    }

    Geometry geometry{};
    LightSampler light_sampler{};
    std::vector<Light> lights{};

    std::vector<Texture> textures{};
    std::unordered_map<std::string, TextureId> builtin_textures{};

    ChunkAllocator<> material_allocator{};
    std::vector<Material> materials{};

    ChunkAllocator<> spectrum_allocator{};
    std::unordered_map<std::string, Spectrum> builtin_spectra{};

    std::optional<Envmap> envmap{};

    SceneAttribs attribs{};

    Scene(const Scene &other) = delete;

    Scene(Scene &&other) noexcept
        : geometry(std::move(other.geometry)),
          light_sampler(std::move(other.light_sampler)), lights(std::move(other.lights)),
          textures(std::move(other.textures)),
          material_allocator(std::move(other.material_allocator)),
          materials(std::move(other.materials)),
          spectrum_allocator(std::move(other.spectrum_allocator)),
          envmap(std::move(other.envmap)), attribs(std::move(other.attribs)) {}

    Scene &
    operator=(const Scene &other) = delete;

    Scene &
    operator=(Scene &&other) noexcept {
        if (this == &other)
            return *this;
        geometry = std::move(other.geometry);
        light_sampler = std::move(other.light_sampler);
        lights = std::move(other.lights);
        textures = std::move(other.textures);
        material_allocator = std::move(other.material_allocator);
        spectrum_allocator = std::move(other.spectrum_allocator);
        materials = std::move(other.materials);
        envmap = std::move(other.envmap);
        attribs = other.attribs;
        return *this;
    }
};

#endif // PT_SCENE_H
