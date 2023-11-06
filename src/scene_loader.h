#ifndef PT_SCENE_LOADER_H
#define PT_SCENE_LOADER_H

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

#include <pugixml.hpp>

#include "utils/numtypes.h"

// Circular dependencies...
class RenderContext;
#include "envmap.h"
#include "render_context_common.h"
#include "texture.h"

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

    std::optional<SceneAttribs> load_scene_attribs();
    void load_scene(RenderContext *rc);

private:
    static void load_rectangle(pugi::xml_node shape, u32 mat_id, const mat4 &transform,
                               cuda::std::optional<u32>, RenderContext *rc);
    static void load_cube(pugi::xml_node shape_node, u32 mat_id, const mat4 &transform,
                          cuda::std::optional<u32>, RenderContext *rc);
    void load_obj(pugi::xml_node shape_node, u32 mat_id, const mat4 &transform,
                  cuda::std::optional<u32>, RenderContext *rc);
    void load_materials(pugi::xml_node scene_node, RenderContext *rc);
    static vec3 parse_rgb(const std::string &str);
    static mat4 parse_transform(pugi::xml_node transform_node);
    static u32 load_emitter(pugi::xml_node emitter_node, RenderContext *rc);
    void load_shapes(RenderContext *rc, const pugi::xml_node &scene);

    std::string scene_base_path;
    pugi::xml_document doc;
    std::unordered_map<std::string, u32> materials;
};

#endif // PT_SCENE_LOADER_H
