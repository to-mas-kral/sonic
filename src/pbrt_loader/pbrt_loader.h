#ifndef PBRT_LOADER_H
#define PBRT_LOADER_H
#include "lexer.h"
#include "param.h"

#include <filesystem>
#include <fstream>
#include <vector>

#include "../math/transform.h"
#include "../scene/scene.h"
#include "../utils/basic_types.h"

struct AttributeState {
    SquareMatrix4 ctm = SquareMatrix4::identity();
    bool reverse_orientation = false;
    std::optional<Emitter> emitter{};
    MaterialId material{0};
    ColorSpace color_space{ColorSpace::sRGB};
};

// Only used in a private function... not sure if it could be moved somewhere else ?
struct RoughnessDescription {
    enum class RoughnessType {
        Isotropic,
        Anisotropic,
    } type;

    TextureId roughness;
    TextureId uroughness;
    TextureId vroughness;
};

// TODO: consider a custom allocator in the future
class PbrtLoader {
public:
    explicit
    PbrtLoader(const std::filesystem::path &file_path);

    explicit
    PbrtLoader(std::istream &istream);

    void
    load_scene(Scene &sc);

private:
    void
    load_screenwide_options(Scene &sc);

    void
    load_camera(Scene &sc);

    void
    load_film(Scene &sc);

    void
    load_integrator(Scene &sc);

    void
    load_identity();

    void
    load_translate();

    void
    load_scale();

    void
    load_rotate();

    void
    load_lookat();

    void
    load_transform();

    void
    load_concat_transform();

    void
    load_scene_description(Scene &sc);

    void
    attribute_begin();

    void
    attribute_end();

    void
    load_shape(Scene &sc);

    void
    load_trianglemesh(Scene &sc, ParamsList &params) const;

    void
    load_plymesh(Scene &sc, ParamsList &params) const;

    void
    load_sphere(Scene &sc, ParamsList &params) const;

    void
    area_light_source(Scene &sc);

    void
    load_material(Scene &sc);

    void
    load_make_named_material(Scene &sc);

    Material
    parse_material_description(Scene &sc, const std::string &type, ParamsList &params);

    RoughnessDescription
    parse_material_roughness(Scene &sc, ParamsList &params);

    Material
    parse_coateddiffuse_material(Scene &sc, ParamsList &params);

    Material
    parse_diffuse_material(Scene &sc, ParamsList &params);

    Material
    parse_dielectric_material(Scene &sc, ParamsList &params);

    Material
    parse_conductor_material(Scene &sc, ParamsList &params);

    TextureId
    get_texture_id_or_default(Scene &sc, ParamsList &params, const std::string &name,
                              const std::string &default_tex);

    TextureId
    parse_inline_spectrum_texture(const Param &param, Scene &sc);

    TextureId
    parse_inline_float_texture(const Param &param, Scene &sc);

    void
    load_named_material();

    void
    load_texture(Scene &sc);

    Lexeme
    expect(LexemeType lt);

    // I want to test the param list parsing logic, but pulling it out into
    // it's own class seems to be more trouble than it's worth
#ifdef TEST_PUBLIC
public:
#else
private:
#endif
    ParamsList
    parse_param_list();

    Param
    parse_param(const std::string_view &type, std::string &&name);

    Param
    parse_spectrum_param(std::string &&name);

    template <typename E, typename P>
    Param
    parse_value_list(P parse, const bool might_be_list, std::string &&name) {
        if (!might_be_list) {
            auto elem = parse();
            return Param(std::move(name), std::move(elem));
        }

        // TODO: default size (maybe 8) might be good
        std::vector<E> values{};

        while (lexer.peek().type != LexemeType::CloseBracket) {
            const auto elem = parse();
            values.push_back(elem);
        }

        if (values.size() == 1) {
            return Param(std::move(name), std::move(values[0]));
        } else if (values.size() > 1) {
            return Param(std::move(name), std::move(values));
        } else {
            throw std::runtime_error(
                fmt::format("Param value list doesn't have any values"));
        }
    }

    i32
    parse_int();

    f32
    parse_float();

    vec2
    parse_point2();

    vec2
    parse_vec2();

    point3
    parse_point3();

    vec3
    parse_vec3();

    vec3
    parse_normal();

    tuple3
    parse_rgb();

    bool
    parse_bool();

    std::string
    parse_quoted_string();

    // TODO: gonna require a lexer stack at some point to support includes
    std::vector<char> buf = std::vector<char>(8192 * 1000);
    std::ifstream file_stream{};
    std::filesystem::path file_path{};
    std::filesystem::path file_directory{};
    Lexer lexer;

    // TODO: could choose more efficient map
    std::unordered_map<std::string, MaterialId> materials{};
    std::unordered_map<std::string, TextureId> textures{};
    std::vector<AttributeState> astates{};
    AttributeState current_astate{};
};

#endif // PBRT_LOADER_H
