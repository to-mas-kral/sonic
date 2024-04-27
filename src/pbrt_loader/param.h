#ifndef PARAM_H
#define PARAM_H

#include <vector>

#include "../math/vecmath.h"
#include "../utils/basic_types.h"

#include <spdlog/spdlog.h>

struct TextureValue {
    std::string str;
};

enum class ValueType {
    Int,
    Float,
    Point2,
    Vector2,
    Point3,
    Vector3,
    Normal,
    Spectrum,
    Rgb,
    // Blackbody,
    Bool,
    String,
    Texture,
};

enum class ParamType {
    Simple,
    Single,
    List,
};

struct Param {
    explicit
    Param(std::string &&name)
        : type{ParamType::Simple}, value_type(ValueType::Int), name{name}, inner{0} {}

    Param(std::string &&name, const i32 value)
        : type{ParamType::Single}, value_type{ValueType::Int}, name{name}, inner{value} {}

    Param(std::string &&name, const f32 value)
        : type{ParamType::Single}, value_type{ValueType::Float}, name{name},
          inner{value} {}

    Param(std::string &&name, const vec2 value)
        : type{ParamType::Single}, value_type{ValueType::Vector2}, name{name},
          inner{value} {}

    Param(std::string &&name, const point3 value)
        : type{ParamType::Single}, value_type{ValueType::Point3}, name{name},
          inner{value} {}

    Param(std::string &&name, const vec3 value)
        : type{ParamType::Single}, value_type{ValueType::Vector3}, name{name},
          inner{value} {}

    Param(std::string &&name, const norm_vec3 value)
        : type{ParamType::Single}, value_type{ValueType::Normal}, name{name},
          inner{value} {}

    Param(std::string &&name, const tuple3 value)
        : type{ParamType::Single}, value_type{ValueType::Rgb}, name{name}, inner{value} {}

    Param(std::string &&name, const bool value)
        : type{ParamType::Single}, value_type{ValueType::Bool}, name{name}, inner{value} {
    }

    Param(std::string &&name, std::string &&value)
        : type{ParamType::Single}, value_type{ValueType::String}, name{name},
          inner{value} {}

    Param(std::string &&name, TextureValue &&value)
        : type{ParamType::Single}, value_type{ValueType::Texture}, name{name},
          inner{value.str} {}

    Param(std::string &&name, std::vector<i32> &&value)
        : type{ParamType::List}, value_type{ValueType::Int}, name{name}, inner{value} {}

    Param(std::string &&name, std::vector<f32> &&value)
        : type{ParamType::List}, value_type{ValueType::Float}, name{name}, inner{value} {}

    Param(std::string &&name, std::vector<vec2> &&value)
        : type{ParamType::List}, value_type{ValueType::Vector2}, name{name},
          inner{value} {}

    Param(std::string &&name, std::vector<point3> &&value)
        : type{ParamType::List}, value_type{ValueType::Point3}, name{name}, inner{value} {
    }

    Param(std::string &&name, std::vector<vec3> &&value)
        : type{ParamType::List}, value_type{ValueType::Vector3}, name{name},
          inner{value} {}

    Param(std::string &&name, std::vector<norm_vec3> &&value)
        : type{ParamType::List}, value_type{ValueType::Normal}, name{name}, inner{value} {
    }

    ParamType type;
    // Don't care about the value type for Simple Params
    ValueType value_type;
    std::string name;

    std::variant<i32, f32, vec2, point3, vec3, norm_vec3, tuple3, bool, std::string,
                 std::vector<i32>, std::vector<f32>, std::vector<vec2>,
                 std::vector<point3>, std::vector<vec3>, std::vector<norm_vec3>>
        inner;

    Param(const Param &other) = delete;

    Param &
    operator=(const Param &other) = delete;

    Param &
    operator=(Param &&other) noexcept {
        if (this == &other)
            return *this;
        type = other.type;
        value_type = other.value_type;
        name = std::move(other.name);
        inner = std::move(other.inner);
        return *this;
    }

    Param(Param &&other) noexcept
        : type(other.type), value_type(other.value_type), name(std::move(other.name)),
          inner{std::move(other.inner)} {
        other.type = ParamType::Simple;
        other.value_type = ValueType::Int;
    }
};

struct ParamsList {
    void
    push(Param &&param) {
        params.push_back(std::move(param));
    }

    std::vector<Param> params{};
};

#endif // PARAM_H
