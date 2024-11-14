#ifndef PT_SCENE_H
#define PT_SCENE_H

#include "../geometry/geometry_container.h"
#include "../geometry/instance_id.h"
#include "../integrator/intersection.h"
#include "../integrator/light_sampler.h"
#include "../materials/material.h"
#include "../math/aabb.h"
#include "envmap.h"
#include "light.h"
#include "scene_attribs.h"
#include "texture.h"

#include <deque>

// TODO: redo scene initialization when refactoring pbrt_loader...
struct Scene {
    Scene();

    void
    set_envmap(Envmap &&a_envmap);

    void
    add_mesh(const MeshParams &mp, std::optional<InstanceId> instance = {});

    void
    add_sphere(const SphereParams &sp, std::optional<InstanceId> instance = {});

    InstanceId
    init_instance();

    void
    add_instanced_instance(InstanceId instance, const SquareMatrix4 &world_from_instance);

    MaterialId
    add_material(const Material &material);

    FloatTexture *
    add_texture(const FloatTexture &texture);

    SpectrumTexture *
    add_texture(const SpectrumTexture &texture);

    Image *
    make_or_get_image(const std::filesystem::path &path);

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
    set_scene_bounds(const AABB &bounds);

    void
    init_light_sampler();

    Scene(const Scene &other) = delete;

    Scene(Scene &&other) noexcept = default;

    Scene &
    operator=(const Scene &other) = delete;

    Scene &
    operator=(Scene &&other) = default;

    ~
    Scene() = default;

    std::optional<LightSample>
    sample_lights(f32 sample, const vec3 &shape_rng, const SampledLambdas &lambdas,
                  const Intersection &its) const;

    // TODO: encapsulate
    GeometryContainer geometry_container{};
    LightSampler light_sampler;
    SceneAttribs attribs{};
    std::vector<Light> lights;
    std::deque<Material> materials;
    std::unique_ptr<Envmap> envmap{nullptr};

    std::unordered_map<std::string, FloatTexture *> builtin_float_textures;
    std::unordered_map<std::string, SpectrumTexture *> builtin_spectrum_textures;

private:
    std::deque<FloatTexture> float_textures;
    std::deque<SpectrumTexture> spectrum_textures;

    std::unordered_map<std::filesystem::path, Image *> images_by_name;
    // Store the images in a deque, because it provides pointer stability
    std::deque<Image> images;

    std::unordered_map<std::string, Spectrum> builtin_spectra;

    AABB m_bounds{vec3(0.F), vec3(0.F)};
};

#endif // PT_SCENE_H
