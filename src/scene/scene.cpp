#include "scene.h"

#include "../color/spectral_data.h"
#include "../integrator/intersection.h"

Scene::
Scene() {
    builtin_spectra.insert({"metal-Au-eta", Spectrum(AU_ETA)});
    builtin_spectra.insert({"metal-Au-k", Spectrum(AU_K)});
    builtin_spectra.insert({"metal-Al-eta", Spectrum(AL_ETA)});
    builtin_spectra.insert({"metal-Al-k", Spectrum(AL_K)});
    builtin_spectra.insert({"metal-Ag-eta", Spectrum(AG_ETA)});
    builtin_spectra.insert({"metal-Ag-k", Spectrum(AG_K)});
    builtin_spectra.insert({"metal-Cu-eta", Spectrum(CU_ETA)});
    builtin_spectra.insert({"metal-Cu-k", Spectrum(CU_K)});

    builtin_spectra.insert({"metal-TiO2-eta", Spectrum(TIO2_ETA)});
    builtin_spectra.insert({"metal-TiO2-k", Spectrum(TIO2_K)});

    builtin_spectra.insert({"glass-BK7", Spectrum(GLASS_BK7_ETA)});
    builtin_spectra.insert({"glass-BAF10", Spectrum(GLASS_BAF10_ETA)});
    builtin_spectra.insert({"glass-F11", Spectrum(GLASS_F11_ETA)});

    for (const auto &spectrum : builtin_spectra) {
        builtin_spectrum_textures.insert(
            {spectrum.first, add_texture(SpectrumTexture(spectrum.second))});
    }

    builtin_spectrum_textures.insert(
        {"reflectance",
         add_texture(SpectrumTexture(RgbSpectrum::from_rgb(tuple3(0.5F))))});
    builtin_float_textures.insert({"roughness", add_texture(FloatTexture(0.F))});
    builtin_spectrum_textures.insert(
        {"eta-dielectric",
         add_texture(SpectrumTexture(Spectrum(ConstantSpectrum(1.5F))))});
    builtin_spectrum_textures.insert(
        {"eta-conductor",
         add_texture(SpectrumTexture(builtin_spectra.at("metal-Cu-eta")))});
    builtin_spectrum_textures.insert(
        {"k-conductor",
         add_texture(SpectrumTexture(builtin_spectra.at("metal-Cu-k")))});
}

MaterialId
Scene::add_material(const Material &material) {
    const u32 mat_id = materials.size();
    materials.push_back(material);
    return MaterialId(mat_id);
}

FloatTexture *
Scene::add_texture(const FloatTexture &texture) {
    float_textures.push_back(texture);
    return &float_textures.back();
}

SpectrumTexture *
Scene::add_texture(const SpectrumTexture &texture) {
    spectrum_textures.push_back(texture);
    return &spectrum_textures.back();
}

Image *
Scene::make_or_get_image(const std::filesystem::path &path) {
    if (images_by_name.contains(path)) {
        return images_by_name.at(path);
    } else {
        auto img = Image::from_filepath(path);
        images.push_back(std::move(img));
        auto *const new_ptr = &images.back();
        images_by_name.insert({path, new_ptr});
        return new_ptr;
    }
}

void
Scene::set_scene_bounds(const AABB &bounds) {
    if (envmap) {
        envmap->set_bounds(bounds);
    }

    m_bounds = bounds;
}

void
Scene::init_light_sampler() {
    light_sampler = LightSampler(lights, geometry_container);
}

std::optional<LightSample>
Scene::sample_lights(const f32 sample, const vec3 &shape_rng,
                     const SampledLambdas &lambdas, const Intersection &its) const {
    const auto index_sample = light_sampler.sample(lights, sample);
    if (!index_sample.has_value()) {
        return {};
    }
    return index_sample->light->sample(index_sample->pdf, shape_rng, lambdas, its,
                                       geometry_container);
}

void
Scene::add_mesh(const MeshParams &mp, const std::optional<InstanceId> instance) {
    const auto next_mesh_id = geometry_container.get_next_shape_index(ShapeType::Mesh);
    std::optional<u32> lights_start_id = {};

    if (mp.emitter.has_value() && instance.has_value()) {
        throw std::runtime_error("Instanced lights arent implemented yet");
    }

    if (mp.emitter.has_value()) {
        lights_start_id = std::optional<u32>(lights.size());

        for (u32 i = 0; i < mp.num_indices / 3; i++) {
            const auto si = ShapeIndex{
                .type = ShapeType::Mesh, .index = next_mesh_id, .triangle_index = i};

            lights.emplace_back(ShapeLight(si, mp.emitter.value()));
        }
    }

    geometry_container.add_mesh(mp, lights_start_id, instance);
}

void
Scene::add_sphere(const SphereParams &sp, const std::optional<InstanceId> instance) {
    const auto next_sphere_id =
        geometry_container.get_next_shape_index(ShapeType::Sphere);
    std::optional<u32> light_id = {};

    if (sp.emitter.has_value() && instance.has_value()) {
        throw std::runtime_error("Instanced lights arent implemented yet");
    }

    if (sp.emitter.has_value()) {
        light_id = std::optional<u32>(lights.size());
        const auto si = ShapeIndex{.type = ShapeType::Sphere, .index = next_sphere_id};

        lights.emplace_back(ShapeLight(si, sp.emitter.value()));
    }

    geometry_container.add_sphere(sp, light_id, instance);
}

InstanceId
Scene::init_instance() {
    return geometry_container.init_instance();
}

void
Scene::add_instanced_instance(const InstanceId instance,
                              const SquareMatrix4 &world_from_instance) {
    geometry_container.add_instanced_instance(instance, world_from_instance);
}

void
Scene::set_envmap(Envmap &&a_envmap) {
    if (envmap != nullptr) {
        throw std::runtime_error("Setting envmap when already set");
    }

    envmap = std::make_unique<Envmap>(std::move(a_envmap));
    const auto envmap_light_id = lights.size();
    lights.emplace_back(EnvmapLight(envmap.get()));
    envmap->set_light_id(envmap_light_id);
}
