#ifndef PT_SCENE_LOADER_H
#define PT_SCENE_LOADER_H

#include <filesystem>
#include <string>
#include <unordered_map>

#include <pugixml.hpp>

#include "../math/vecmath.h"
#include "../scene/envmap.h"
#include "../scene/scene.h"
#include "../scene/texture.h"
#include "../utils/basic_types.h"

struct SceneAttribs {
    u32 resx = 1280;
    u32 resy = 720;
    f32 fov = 30.f;
    u32 max_depth = 0;
    mat4 camera_to_world = mat4::identity();
};

/// Loader for Mitsuba's scene format:
/// https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#scene-xml-file-format
/// Only supports a very limited subset of features...
class SceneLoader {
public:
    SceneLoader() : materials(std::unordered_map<std::string, u32>{}){};

    explicit SceneLoader(std::string scene_path) {
        pugi::xml_parse_result result = doc.load_file(scene_path.data());
        if (!result) {
            throw std::runtime_error(
                fmt::format("Couldn't find scene XML file: {}", scene_path));
        }

        auto path = std::filesystem::path(scene_path);
        scene_base_path = path.parent_path();
    }

    std::optional<SceneAttribs>
    load_scene_attribs();
    void
    load_scene(Scene *sc);

private:
    static void
    load_rectangle(pugi::xml_node shape, u32 mat_id, const mat4 &transform,
                   Option<Emitter>, Scene *sc);

    static void
    load_cube(pugi::xml_node shape_node, u32 mat_id, const mat4 &transform,
              Option<Emitter>, Scene *sc);

    void
    load_obj(pugi::xml_node shape_node, u32 mat_id, const mat4 &transform,
             Option<Emitter>, Scene *sc);

    void
    load_sphere(pugi::xml_node node, u32 id, mat4 mat_1, Option<Emitter> emitter_id,
                Scene *sc);

    void
    load_materials(pugi::xml_node scene_node, Scene *sc);

    static tuple3
    parse_tuple3(const std::string &str);

    static mat4
    parse_transform(pugi::xml_node transform_node);

    static Emitter
    load_emitter(pugi::xml_node emitter_node, Scene *sc);

    void
    load_shapes(Scene *sc, const pugi::xml_node &scene);

    std::tuple<Material, std::string>
    load_material(Scene *scene, pugi::xml_node &bsdf);

    Material
    load_diffuse_material(Scene *sc, const pugi::xml_node &bsdf);

    Material
    load_plastic_material(Scene *sc, const pugi::xml_node &bsdf) const;

    Material
    load_roughp_lastic_material(Scene *sc, const pugi::xml_node &bsdf) const;

    Material
    load_conductor_material(const pugi::xml_node &bsdf) const;

    Material
    load_dielectric_material(const pugi::xml_node &bsdf) const;

    Material
    load_roughconductor_material(const pugi::xml_node &bsdf) const;

    static mat4
    parse_transform_matrix(const pugi::xml_node &matrix_node);

    static mat4
    parse_transform_rotate(const pugi::xml_node &transform_node);

    u32
    load_texture(Scene *sc, const pugi::xml_node &texture_node) const;

    std::string scene_base_path;
    pugi::xml_document doc;
    std::unordered_map<std::string, u32> materials;
};

#endif // PT_SCENE_LOADER_H
