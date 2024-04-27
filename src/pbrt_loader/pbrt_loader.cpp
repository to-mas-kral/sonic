#include "pbrt_loader.h"

#include <charconv>

Lexer
init_lexer(const std::filesystem::path &file_path, std::vector<char> &buf,
           std::ifstream &file_stream) {
    file_stream.rdbuf()->pubsetbuf(buf.data(), buf.size());
    file_stream.open(file_path);

    return Lexer(&file_stream);
}

PbrtLoader::
PbrtLoader(const std::filesystem::path &file_path)
    : file_path{file_path}, file_directory{file_path.parent_path()},
      lexer{init_lexer(file_path, buf, file_stream)} {}

PbrtLoader::
PbrtLoader(std::istream &istream)
    : lexer{Lexer(&istream)} {}

void
PbrtLoader::load_scene(Scene &sc) {
    load_screenwide_options(sc);
}

void
PbrtLoader::load_screenwide_options(Scene &sc) {
    const auto directive = expect(LexemeType::String);

    while (true) {
        if (directive.src == "Option") {
            spdlog::info("'Option' ignored");
        } else if (directive.src == "Camera") {
            load_camera(sc);
        } else if (directive.src == "Sampler") {
            spdlog::info("'Sampler' ignored");
        } else if (directive.src == "ColorSpace") {
            spdlog::info("'ColorSpace' ignored");
        } else if (directive.src == "Film") {
            throw std::runtime_error("Film unimplemented");
        } else if (directive.src == "PixelFilter") {
            spdlog::info("'PixelFilter' ignored");
        } else if (directive.src == "Integrator") {
            spdlog::info("'Integrator' ignored");
        } else if (directive.src == "Accelerator") {
            spdlog::info("'Accelerator' ignored");
        } else if (directive.src == "WorldBegin") {
            break;
        } else if (directive.src == "MakeNamedMedium") {
            spdlog::info("'MakeNamedMedium' ignored");
        } else if (directive.src == "MediumInterface") {
            spdlog::info("'MediumInterface' ignored");
        } else if (directive.src == "Identity") {
            throw std::runtime_error("Identity unimplemented");
        } else if (directive.src == "Translate") {
            throw std::runtime_error("Translate unimplemented");
        } else if (directive.src == "Scale") {
            throw std::runtime_error("Scale unimplemented");
        } else if (directive.src == "Rotate") {
            throw std::runtime_error("Rotate unimplemented");
        } else if (directive.src == "LookAt") {
            throw std::runtime_error("LookAt unimplemented");
        } else if (directive.src == "CoordinateSystem") {
            throw std::runtime_error("CoordinateSystem unimplemented");
        } else if (directive.src == "CoordSysTransform") {
            throw std::runtime_error("CoordSysTransform unimplemented");
        } else if (directive.src == "Transform") {
            throw std::runtime_error("Transform unimplemented");
        } else if (directive.src == "ConcatTransform") {
            throw std::runtime_error("ConcatTransform unimplemented");
        } else {
            throw std::runtime_error(
                fmt::format("Unknown directive: '{}'", directive.src));
        }
    }
}

void
PbrtLoader::load_camera(Scene &sc) {
    /*let mut cam = Camera {
        camera_from_world_transform: self.gstate.ctm,
        ..Camera::default()
    };

    let mut params = self.parse_param_list()?;

    // TODO: not great parsing

    let typ = params.expect_simple()?;
    match typ {
        "orthographic" => cam.typ = CameraTyp::Orthographic,
        "perspective" => cam.typ = CameraTyp::Perspective,
        "realistic" => cam.typ = CameraTyp::Realistic,
        "spherical" => cam.typ = CameraTyp::Spherical,
        cam => return Err(eyre!("Unkown camera type: '{}'", cam)),
    };

    for p in params.params() {
        match (p.name, &p.value) {
            ("fov", ListParamValue::Single(Value::Float(fov))) => {
                cam.fov = *fov;
            }
            p => return Err(eyre!("Wrong Camera parameter: '{:?}'", p)),
        }
    }

    Ok(cam)*/

    const auto params = parse_param_list();
}

ParamsList
PbrtLoader::parse_param_list() {
    ParamsList params{};

    // The opening quotes of a parameter
    while (lexer.peek().type == LexemeType::Quotes) {
        lexer.next();

        auto type_or_param = expect(LexemeType::String);

        if (lexer.peek().type == LexemeType::Quotes) {
            // A simple parameter
            lexer.next();

            auto param = Param(std::move(type_or_param.src));
            params.push(std::move(param));

            continue;
        }

        // type_or_param is a type
        auto name = expect(LexemeType::String);
        expect(LexemeType::Quotes);

        auto has_brackets = false;
        if (lexer.peek().type == LexemeType::OpenBracket) {
            has_brackets = true;
            lexer.next();
        }

        auto param = parse_param(type_or_param.src, std::move(name.src), has_brackets);
        params.push(std::move(param));

        if (has_brackets) {
            expect(LexemeType::CloseBracket);
        }
    }

    return params;
}

Param
PbrtLoader::parse_param(const std::string_view &type, std::string &&name,
                        const bool might_be_list) {
    if (type == "integer") {
        return parse_value_list<i32>([this] { return parse_int(); }, might_be_list,
                                     std::move(name));
    } else if (type == "float") {
        return parse_value_list<f32>([this] { return parse_float(); }, might_be_list,
                                     std::move(name));
    } else if (type == "point2") {
        return parse_value_list<vec2>([this] { return parse_vec2(); }, might_be_list,
                                      std::move(name));
    } else if (type == "vector2") {
        return parse_value_list<vec2>([this] { return parse_vec2(); }, might_be_list,
                                      std::move(name));
    } else if (type == "point3") {
        return parse_value_list<point3>([this] { return parse_point3(); }, might_be_list,
                                        std::move(name));
    } else if (type == "vector3") {
        return parse_value_list<vec3>([this] { return parse_vec3(); }, might_be_list,
                                      std::move(name));
    } else if (type == "normal" || type == "normal3") {
        return parse_value_list<norm_vec3>([this] { return parse_normal(); },
                                           might_be_list, std::move(name));
    } else if (type == "spectrum") {
        throw std::runtime_error("spectrum param unimplemented");
    } else if (type == "rgb") {
        const auto rgb = parse_rgb();
        return Param(std::move(name), rgb);
    } else if (type == "blackbody") {
        throw std::runtime_error("blackbody param unimplemented");
    } else if (type == "bool") {
        const auto _bool = parse_bool();
        return Param(std::move(name), _bool);
    } else if (type == "string") {
        auto str = parse_quoted_string();
        return Param(std::move(name), std::move(str));
    } else if (type == "texture") {
        auto str = parse_quoted_string();
        return Param(std::move(name), TextureValue{.str = std::move(str)});
    } else {
        throw std::runtime_error(fmt::format("Unknown param type: '{}'", type));
    }
}

i32
PbrtLoader::parse_int() {
    const auto num = expect(LexemeType::Num);
    i32 _int{};
    const auto res =
        std::from_chars(num.src.data(), num.src.data() + num.src.size(), _int);

    if (res.ec == std::errc::invalid_argument ||
        res.ec == std::errc::result_out_of_range) {
        throw std::runtime_error(fmt::format("Error decoding an int: '{}'", num.src));
    }

    return _int;
}

// TODO: could use something like "fast_float" by Lemire
f32
PbrtLoader::parse_float() {
    const auto num = expect(LexemeType::Num);
    f32 _float{};
    const auto res =
        std::from_chars(num.src.data(), num.src.data() + num.src.size(), _float);

    if (res.ec == std::errc::invalid_argument ||
        res.ec == std::errc::result_out_of_range) {
        throw std::runtime_error(fmt::format("Error decoding an int: '{}'", num.src));
    }

    return _float;
}

vec2
PbrtLoader::parse_point2() {
    const auto x = parse_float();
    const auto y = parse_float();
    return vec2(x, y);
}

vec2
PbrtLoader::parse_vec2() {
    const auto x = parse_float();
    const auto y = parse_float();
    return vec2(x, y);
}

point3
PbrtLoader::parse_point3() {
    const auto x = parse_float();
    const auto y = parse_float();
    const auto z = parse_float();
    return point3(x, y, z);
}

vec3
PbrtLoader::parse_vec3() {
    const auto x = parse_float();
    const auto y = parse_float();
    const auto z = parse_float();
    return vec3(x, y, z);
}

norm_vec3
PbrtLoader::parse_normal() {
    const auto x = parse_float();
    const auto y = parse_float();
    const auto z = parse_float();
    return norm_vec3(x, y, z);
}

tuple3
PbrtLoader::parse_rgb() {
    const auto x = parse_float();
    const auto y = parse_float();
    const auto z = parse_float();
    return tuple3(x, y, z);
}

bool
PbrtLoader::parse_bool() {
    const auto lex = expect(LexemeType::String);
    if (lex.src == "true") {
        return true;
    } else if (lex.src == "false") {
        return false;
    } else {
        throw std::runtime_error(fmt::format("Invalid bool value: '{}'", lex.src));
    }
}

std::string
PbrtLoader::parse_quoted_string() {
    expect(LexemeType::Quotes);
    const auto lex = expect(LexemeType::String);
    expect(LexemeType::Quotes);

    return lex.src;
}

Lexeme
PbrtLoader::expect(const LexemeType lt) {
    const auto lex = lexer.next();
    if (lex.type != lt) {
        throw std::runtime_error("wrong lexeme type");
    } else {
        return lex;
    }
}
