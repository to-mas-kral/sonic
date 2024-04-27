#ifndef PT_SCENE_H
#define PT_SCENE_H

#include "../geometry/geometry.h"
#include "../integrator/light_sampler.h"
#include "../materials/material.h"
#include "envmap.h"
#include "light.h"
#include "scene_attribs.h"
#include "texture.h"

struct Scene {
    Scene() = default;

    ~
    Scene();

    void
    set_envmap(Envmap &&a_envmap) {
        envmap = std::move(a_envmap);
        has_envmap = true;
    }

    void
    add_mesh(MeshParams mp);
    void
    add_sphere(SphereParams sp);

    u32
    add_material(const Material &material);

    u32
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

    ChunkAllocator<> material_allocator{};
    ChunkAllocator<> spectrum_allocator{};
    std::vector<Material> materials{};

    Envmap envmap{};
    bool has_envmap = false;

    SceneAttribs attribs{};
};

#endif // PT_SCENE_H
