#ifndef PT_SCENE_LOADER_H
#define PT_SCENE_LOADER_H

#include <filesystem>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>

#include <pugixml.hpp>

#include "utils/basic_types.h"
#include "math/vecmath.h"
#include "envmap.h"
#include "scene.h"
#include "texture.h"

struct SceneAttribs {
    u32 resx{};
    u32 resy{};
    f32 fov{};
    mat4 camera_to_world = mat4::identity();
};

/// Loader for Mitsuba's scene format:
/// https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html#scene-xml-file-format
/// Only supports a very limited subset of features...
class SceneLoader {
public:
    SceneLoader()
        : materials(std::unordered_map<std::string, u32>{}){

          };

    explicit SceneLoader(std::string scene_path) {
        pugi::xml_parse_result result = doc.load_file(scene_path.data());
        if (!result) {
            throw;
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
                   COption<Emitter>, Scene *sc);
    static void
    load_cube(pugi::xml_node shape_node, u32 mat_id, const mat4 &transform,
              COption<Emitter>, Scene *sc);
    void
    load_obj(pugi::xml_node shape_node, u32 mat_id, const mat4 &transform,
             COption<Emitter>, Scene *sc);
    void
    load_sphere(pugi::xml_node node, u32 id, mat4 mat_1, COption<Emitter> emitter_id,
                Scene *sc);
    void
    load_materials(pugi::xml_node scene_node, Scene *sc);
    static vec3
    parse_rgb(const std::string &str);
    static mat4
    parse_transform(pugi::xml_node transform_node);
    static Emitter
    load_emitter(pugi::xml_node emitter_node, Scene *sc);
    void
    load_shapes(Scene *sc, const pugi::xml_node &scene);

    std::string scene_base_path;
    pugi::xml_document doc;
    std::unordered_map<std::string, u32> materials;
    std::tuple<Material, std::string>
    load_material(Scene *sc, pugi::xml_node &bsdf);
};

#endif // PT_SCENE_LOADER_H
