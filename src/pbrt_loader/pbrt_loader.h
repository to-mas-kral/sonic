#ifndef PBRT_LOADER_H
#define PBRT_LOADER_H
#include "lexer.h"
#include "param.h"

#include <filesystem>
#include <fstream>
#include <map>
#include <vector>

#include "../math/transform.h"
#include "../scene/scene.h"
#include "../utils/basic_types.h"

struct AttributeState {
    SquareMatrix4 ctm = SquareMatrix4::identity();
    bool reverse_orientation = false;
    // area light source
    // material
    // color space
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

    Lexeme
    expect(LexemeType lt);

    // I want to test the param list parsing logic, but pulling it out into
    // it's own class seems to be more trouble than it's worth
#ifdef TEST_PUBLIC
public:
#endif
    ParamsList
    parse_param_list();

private:
    Param
    parse_param(const std::string_view &type, std::string &&name, bool might_be_list);

    template <typename E, typename P>
    Param
    parse_value_list(P parse, const bool might_be_list, std::string &&name) {
        if (!might_be_list) {
            const auto elem = parse();
            return Param(std::move(name), elem);
        }

        std::vector<E> values{};

        while (lexer.peek().type != LexemeType::CloseBracket) {
            const auto elem = parse();
            values.push_back(elem);
        }

        if (values.size() == 1) {
            return Param(std::move(name), values[0]);
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

    norm_vec3
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

    std::map<std::string, u32> materials{};
};

#endif // PBRT_LOADER_H
