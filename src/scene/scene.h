#ifndef PT_SCENE_H
#define PT_SCENE_H

#include "../geometry/geometry.h"
#include "../geometry/instance_id.h"
#include "../integrator/light_sampler.h"
#include "../materials/material.h"
#include "envmap.h"
#include "light.h"
#include "scene_attribs.h"
#include "texture.h"

struct Scene {
    Scene();

    ~
    Scene();

    void
    set_envmap(const Envmap &a_envmap) {
        envmap = a_envmap;
    }

    void
    add_mesh(const MeshParams &mp, std::optional<InstanceId> instance = {});

    void
    add_sphere(const SphereParams &sp, std::optional<InstanceId> instance = {});

    InstanceId
    init_instance();

    void
    add_instance(InstanceId instance, const SquareMatrix4 &world_from_instance);

    MaterialId
    add_material(const Material &material);

    FloatTexture *
    add_texture(const FloatTexture &texture);

    SpectrumTexture *
    add_texture(const SpectrumTexture &texture);

    Image *
    get_image(const std::filesystem::path &path);

    template <typename T>
    T *
    get_builtin_texture(const std::string &name) {
        if constexpr (std::is_same<T, FloatTexture>()) {
            return builtin_float_textures.at(name);
        } else if constexpr (std::is_same<T, SpectrumTexture>()) {
            return builtin_spectrum_textures.at(name);
        } else {
            static_assert(false);
        }
    }

    void
    init_light_sampler();

    Option<LightSample>
    sample_lights(f32 sample) {
        return light_sampler.sample(lights, sample);
    }

    Geometry geometry{};
    LightSampler light_sampler{};
    std::vector<Light> lights{};

    std::vector<FloatTexture> float_textures{};
    std::vector<SpectrumTexture> spectrum_textures{};
    std::unordered_map<std::string, FloatTexture *> builtin_float_textures{};
    std::unordered_map<std::string, SpectrumTexture *> builtin_spectrum_textures{};
    ChunkAllocator<> texture_allocator{};
    ChunkAllocator<> image_allocator{};
    std::unordered_map<std::filesystem::path, Image *> images{};

    ChunkAllocator<> material_allocator{};
    std::vector<Material> materials{};

    std::unordered_map<std::string, Spectrum> builtin_spectra{};

    std::optional<Envmap> envmap{};

    SceneAttribs attribs{};

    Scene(const Scene &other) = delete;

    Scene(Scene &&other) noexcept = default;

    Scene &
    operator=(const Scene &other) = delete;

    Scene &
    operator=(Scene &&other) = default;
};

#endif // PT_SCENE_H
