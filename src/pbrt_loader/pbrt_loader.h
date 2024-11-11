#ifndef PBRT_LOADER_H
#define PBRT_LOADER_H
#include "lexer.h"
#include "param.h"
#include "stack_file_stream.h"

#include <filesystem>
#include <vector>

#include "../geometry/instance_id.h"
#include "../math/transform.h"
#include "../scene/scene.h"
#include "../utils/basic_types.h"

struct AttributeState {
    SquareMatrix4 ctm = SquareMatrix4::identity();
    bool reverse_orientation = false;
    std::optional<Emitter> emitter;
    MaterialId material{0};
    ColorSpace color_space{ColorSpace::sRGB};
    std::optional<InstanceId> instance;
};

// Only used in a private function... not sure if it could be moved somewhere else ?
struct RoughnessDescription {
    enum class RoughnessType : u8 {
        Isotropic,
        Anisotropic,
    } type;

    FloatTexture *roughness;
    FloatTexture *uroughness;
    FloatTexture *vroughness;
};

class PbrtLoader {
public:
    explicit
    PbrtLoader(const std::filesystem::path &file_path);

    explicit
    PbrtLoader(const std::string &input);

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
    object_begin(Scene &sc);

    void
    object_end();

    void
    object_instance(Scene &sc);

    void
    load_light_source(Scene &sc);
    void
    normals_reverse_orientation(u32 num_verts, vec3 *normals) const;
    void
    transform_mesh(point3 *pos, u32 num_verts, vec3 *normals) const;

    void
    load_trianglemesh(Scene &sc, ParamsList &params, FloatTexture *alpha) const;

    void
    load_plymesh(Scene &sc, ParamsList &params, FloatTexture *alpha) const;

    void
    load_sphere(Scene &sc, ParamsList &params, FloatTexture *alpha) const;

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
    parse_diffusetransmission_material(Scene &sc, ParamsList &params);

    Material
    parse_dielectric_material(Scene &sc, ParamsList &params);

    Material
    parse_conductor_material(Scene &sc, ParamsList &params);

    template <typename T>
    T *
    get_texture_or_default(Scene &sc, ParamsList &params, const std::string &name,
                           const std::string &default_tex) {
        const auto opt_p = params.get_optional(name);

        if (!opt_p.has_value()) {
            return sc.get_builtin_texture<T>(default_tex);
        }

        const auto *const p = opt_p.value();
        if (p->value_type == ValueType::Texture) {
            const auto &tex_name = std::get<std::string>(p->inner);
            if constexpr (std::is_same<T, FloatTexture>()) {
                return float_textures.at(tex_name);
            } else if constexpr (std::is_same<T, SpectrumTexture>()) {
                return spectrum_textures.at(tex_name);
            } else {
                static_assert(false);
            }
        }

        if constexpr (std::is_same<T, SpectrumTexture>()) {
            return parse_inline_spectrum_texture(*p, sc);
        } else if constexpr (std::is_same<T, FloatTexture>()) {
            return parse_inline_float_texture(*p, sc);
        } else {
            static_assert(false);
        }
    }

    template <typename T>
    std::optional<T *>
    get_texture_opt(Scene &sc, ParamsList &params, const std::string &name) {
        const auto opt_p = params.get_optional(name);

        if (!opt_p.has_value()) {
            return {};
        }

        const auto *const p = opt_p.value();
        if (p->value_type == ValueType::Texture) {
            const auto &tex_name = std::get<std::string>(p->inner);
            if constexpr (std::is_same<T, FloatTexture>()) {
                return float_textures.at(tex_name);
            } else if constexpr (std::is_same<T, SpectrumTexture>()) {
                return spectrum_textures.at(tex_name);
            } else {
                static_assert(false);
            }
        }

        if constexpr (std::is_same<T, SpectrumTexture>()) {
            return parse_inline_spectrum_texture(*p, sc);
        } else if constexpr (std::is_same<T, FloatTexture>()) {
            return parse_inline_float_texture(*p, sc);
        } else {
            static_assert(false);
        }
    }

    template <typename T>
    T *
    get_texture_required(Scene &sc, ParamsList &params, const std::string &name) {
        const auto &p = params.get_required(name);

        if (p.value_type == ValueType::Texture) {
            const auto &tex_name = std::get<std::string>(p.inner);
            if constexpr (std::is_same<T, FloatTexture>()) {
                return float_textures.at(tex_name);
            } else if constexpr (std::is_same<T, SpectrumTexture>()) {
                return spectrum_textures.at(tex_name);
            } else {
                static_assert(false);
            }
        }

        if constexpr (std::is_same<T, SpectrumTexture>()) {
            return parse_inline_spectrum_texture(p, sc);
        } else if constexpr (std::is_same<T, FloatTexture>()) {
            return parse_inline_float_texture(p, sc);
        } else {
            static_assert(false);
        }
    }

    SpectrumTexture *
    parse_inline_spectrum_texture(const Param &param, Scene &sc);

    FloatTexture *
    parse_inline_float_texture(const Param &param, Scene &sc) const;

    void
    load_named_material();
    void
    load_imagemap_texture(Scene &sc, const std::string &name, ParamsList &params,
                          const std::string &type);
    void
    load_scale_texture(Scene &sc, const std::string &name, ParamsList &params,
                       const std::string &type);
    void
    load_mix_texture(Scene &sc, const std::string &name, ParamsList &params,
                     const std::string &type);

    void
    load_constant_texture(Scene &sc, const std::string &name, ParamsList &params,
                          const std::string &type);

    void
    load_texture(Scene &sc);

    void
    include();

    Lexeme
    expect(LexemeType lt, bool accept_any_string = false);

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

        std::vector<E> values{};
        // Reserve 16, so that transformation matrices fit without resizing
        values.reserve(16);

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

    std::filesystem::path file_path;
    std::filesystem::path base_directory;
    StackFileStream stack_file_stream;
    Lexer lexer;

    std::unordered_map<std::string, MaterialId> materials;
    std::unordered_map<std::string, FloatTexture *> float_textures;
    std::unordered_map<std::string, SpectrumTexture *> spectrum_textures;
    std::unordered_map<std::string, InstanceId> instances;
    std::vector<AttributeState> astates;
    AttributeState current_astate{};
};

#endif // PBRT_LOADER_H
