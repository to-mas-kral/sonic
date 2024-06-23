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
            {spectrum.first, add_texture(SpectrumTexture::make(spectrum.second))});
    }

    builtin_spectrum_textures.insert(
        {"reflectance", add_texture(SpectrumTexture::make(RgbSpectrum(tuple3(0.5f))))});
    builtin_float_textures.insert({"roughness", add_texture(FloatTexture::make(0.f))});
    builtin_spectrum_textures.insert(
        {"eta-dielectric",
         add_texture(SpectrumTexture::make(Spectrum(ConstantSpectrum::make(1.5f))))});
    builtin_spectrum_textures.insert(
        {"eta-conductor",
         add_texture(SpectrumTexture::make(builtin_spectra.at("metal-Cu-eta")))});
    builtin_spectrum_textures.insert(
        {"k-conductor",
         add_texture(SpectrumTexture::make(builtin_spectra.at("metal-Cu-k")))});
}

MaterialId
Scene::add_material(const Material &material) {
    const u32 mat_id = materials.size();
    materials.push_back(material);
    return MaterialId(mat_id);
}

FloatTexture *
Scene::add_texture(const FloatTexture &texture) {
    const auto ptr = texture_allocator.allocate<FloatTexture>();
    const auto new_ptr = new (ptr) FloatTexture(texture);
    return new_ptr;
}

SpectrumTexture *
Scene::add_texture(const SpectrumTexture &texture) {
    const auto ptr = texture_allocator.allocate<SpectrumTexture>();
    const auto new_ptr = new (ptr) SpectrumTexture(texture);
    return new_ptr;
}

Image *
Scene::make_or_get_image(const std::filesystem::path &path) {
    if (images.contains(path)) {
        return images.at(path);
    } else {
        const auto img = Image::make(path);
        const auto ptr = image_allocator.allocate<Image>();
        const auto new_ptr = new (ptr) Image(img);
        images.insert({path, new_ptr});
        return new_ptr;
    }
}

void
Scene::set_scene_bounds(const AABB &bounds) const {
    if (envmap) {
        envmap->set_bounds(bounds);
    }
}

void
Scene::init_light_sampler() {
    light_sampler = LightSampler(lights, geometry);
}

Option<LightSample>
Scene::sample_lights(f32 sample, const vec3 &shape_rng, const SampledLambdas &lambdas,
                     const Intersection &its) {
    const auto index_sample = light_sampler.sample(lights, sample);
    return index_sample->light->sample(index_sample->pdf, shape_rng, lambdas, its,
                                       geometry);
}

void
Scene::add_mesh(const MeshParams &mp, const std::optional<InstanceId> instance) {
    const auto next_mesh_id = geometry.get_next_shape_index(ShapeType::Mesh);
    Option<u32> lights_start_id = {};

    if (mp.emitter.has_value() && instance.has_value()) {
        throw std::runtime_error("Instanced lights arent implemented yet");
    }

    if (mp.emitter.has_value()) {
        lights_start_id = Option<u32>(lights.size());

        for (u32 i = 0; i < mp.num_indices / 3; i++) {
            const auto si = ShapeIndex{
                .type = ShapeType::Mesh, .index = next_mesh_id, .triangle_index = i};

            lights.emplace_back(ShapeLight(si, mp.emitter.value()));
        }
    }

    if (instance.has_value()) {
        const auto instance_id = instance.value();
        auto mesh = Mesh(mp, lights_start_id);
        geometry.instances.instanced_objs[instance_id.inner].meshes.meshes.push_back(
            std::move(mesh));
    } else {
        geometry.add_mesh(mp, lights_start_id);
    }
}

void
Scene::add_sphere(const SphereParams &sp, std::optional<InstanceId> instance) {
    const auto next_sphere_id = geometry.get_next_shape_index(ShapeType::Sphere);
    Option<u32> light_id = {};

    if (sp.emitter.has_value() && instance.has_value()) {
        throw std::runtime_error("Instanced lights arent implemented yet");
    }

    if (sp.emitter.has_value()) {
        light_id = Option<u32>(lights.size());
        const auto si = ShapeIndex{.type = ShapeType::Sphere, .index = next_sphere_id};

        lights.emplace_back(ShapeLight(si, sp.emitter.value()));
    }

    // TODO: refactor this...
    if (instance.has_value()) {
        const auto instance_id = instance.value();
        auto &spheres = geometry.instances.instanced_objs[instance_id.inner].spheres;
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

        spheres.alphas.push_back(sp.alpha);

        spheres.num_spheres++;
    } else {
        geometry.add_sphere(sp, light_id);
    }
}

InstanceId
Scene::init_instance() {
    const u32 id = geometry.instances.instanced_objs.size();
    geometry.instances.instanced_objs.emplace_back();
    return InstanceId{id};
}

void
Scene::add_instance(const InstanceId instance, const SquareMatrix4 &world_from_instance) {
    geometry.instances.indices.push_back(instance.inner);
    geometry.instances.world_from_instances.push_back(world_from_instance);
    geometry.instances.wfi_inv_trans.push_back(world_from_instance.inverse().transpose());
}

Scene::~
Scene() {
    for (const auto &[_, img] : images) {
        img->free();
    }
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
