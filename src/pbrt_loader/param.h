#ifndef PARAM_H
#define PARAM_H

#include <spdlog/spdlog.h>
#include <vector>

#include "../math/vecmath.h"
#include "../utils/basic_types.h"

struct TextureValue {
    std::string str;
};

struct SpectrumValue {
    std::string str;
};

struct BlackbodyValue {
    i32 kelvin;
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
    Blackbody,
    Bool,
    String,
    Texture,
};

inline std::string_view
to_string(const ValueType e) {
    using namespace std::literals;

    switch (e) {
    case ValueType::Int:
        return "Int"sv;
    case ValueType::Float:
        return "Float"sv;
    case ValueType::Point2:
        return "Point2"sv;
    case ValueType::Vector2:
        return "Vector2"sv;
    case ValueType::Point3:
        return "Point3"sv;
    case ValueType::Vector3:
        return "Vector3"sv;
    case ValueType::Normal:
        return "Normal"sv;
    case ValueType::Spectrum:
        return "Spectrum"sv;
    case ValueType::Rgb:
        return "Rgb"sv;
    case ValueType::Bool:
        return "Bool"sv;
    case ValueType::String:
        return "String"sv;
    case ValueType::Texture:
        return "Texture"sv;
    default:
        return "unknown"sv;
    }
}

enum class ParamType {
    Simple,
    Single,
    List,
};

struct Param {
    Param()
        : type{ParamType::Simple}, value_type(ValueType::Int), name{std::string{}},
          inner{0} {}

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

    Param(std::string &&name, SpectrumValue &&value)
        : type{ParamType::Single}, value_type{ValueType::Spectrum}, name{name},
          inner{value.str} {}

    Param(std::string &&name, BlackbodyValue value)
        : type{ParamType::Single}, value_type{ValueType::Blackbody}, name{name},
          inner{value.kelvin} {}

    Param(std::string &&name, std::vector<i32> &&value)
        : type{ParamType::List}, value_type{ValueType::Int}, name{name}, inner{value} {}

    Param(std::string &&name, std::vector<f32> &&value,
          const ValueType vt = ValueType::Float)
        : type{ParamType::List}, value_type{vt}, name{name}, inner{value} {}

    Param(std::string &&name, std::vector<vec2> &&value)
        : type{ParamType::List}, value_type{ValueType::Vector2}, name{name},
          inner{value} {}

    Param(std::string &&name, std::vector<point3> &&value)
        : type{ParamType::List}, value_type{ValueType::Point3}, name{name}, inner{value} {
    }

    Param(std::string &&name, std::vector<vec3> &&value)
        : type{ParamType::List}, value_type{ValueType::Vector3}, name{name},
          inner{value} {}

    Param(std::string &&name, std::vector<std::string> &&value)
        : type{ParamType::List}, value_type{ValueType::String}, name{name}, inner{value} {
    }

    bool was_accessed{false};
    ParamType type;
    // Don't care about the value type for Simple Params
    ValueType value_type;
    std::string name;

    std::variant<i32, f32, vec2, point3, vec3, tuple3, bool, std::string,
                 std::vector<i32>, std::vector<f32>, std::vector<vec2>,
                 std::vector<point3>, std::vector<vec3>, std::vector<std::string>>
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
    add(Param &&param) {
        if (params_by_name.contains(param.name)) {
            const auto id = params_by_name.at(param.name);
            if (params[id].value_type == param.value_type) {
                throw std::runtime_error(
                    fmt::format("Param '{}' already present in param list", param.name));
            }
        }

        auto p_index = params.size();

        if (param.type != ParamType::Simple) {
            params_by_name.insert({param.name, p_index});
        }
        params.push_back(std::move(param));
    }

    std::optional<Param *>
    get_optional(const std::string &name, const ValueType vt) {
        if (!params_by_name.contains(name)) {
            return {};
        }

        const auto p_index = params_by_name.at(name);
        auto &p = params[p_index];

        if (p.value_type != vt) {
            throw std::runtime_error(fmt::format("Param '{}' has wrong type: '{}'", name,
                                                 to_string(p.value_type)));
        }

        p.was_accessed = true;
        return &p;
    }

    std::optional<Param *>
    get_optional(const std::string &name) {
        if (!params_by_name.contains(name)) {
            return {};
        }

        const auto p_index = params_by_name.at(name);
        auto &p = params[p_index];

        p.was_accessed = true;
        return &p;
    }

    Param &
    get_required(const std::string &name, const ValueType vt) {
        if (!params_by_name.contains(name)) {
            throw std::runtime_error(fmt::format("Param '{}' is not present", name));
        }

        const auto p_index = params_by_name.at(name);
        auto &p = params[p_index];

        if (p.value_type != vt) {
            throw std::runtime_error(fmt::format("Param '{}' has wrong type: '{}'", name,
                                                 to_string(p.value_type)));
        }

        p.was_accessed = true;
        return p;
    }

    Param &
    get_required(const std::string &name) {
        if (!params_by_name.contains(name)) {
            throw std::runtime_error(fmt::format("Param '{}' is not present", name));
        }

        const auto p_index = params_by_name.at(name);
        auto &p = params[p_index];

        p.was_accessed = true;
        return p;
    }

    Param &
    next_param() {
        if (index < params.size()) {
            auto &p = params.at(index++);
            p.was_accessed = true;
            return p;
        } else {
            throw std::runtime_error("Param list is empty");
        }
    }

    Param &
    expect(const ParamType pt) {
        auto &next = next_param();
        if (next.type != pt) {
            throw std::runtime_error("Wrong Param type");
        }

        next.was_accessed = true;

        return next;
    }

    void
    warn_unused_params(const std::string_view directive) {
        for (const auto &p : params) {
            if (!p.was_accessed) {
                // TODO: could provide better diagnostic by printing the whole params list
                spdlog::warn("Param '{}' was ignored in directive '{}'", p.name,
                             directive);
            }
        }
    }

private:
#ifdef TEST_PUBLIC
public:
#endif
    i32 index = 0;
    std::vector<Param> params{};
    /// This only ontains params with values, not "simple" params like "imagmap",
    /// because those can technically be duplicated
    std::unordered_map<std::string, std::size_t> params_by_name{};
};

#endif // PARAM_H
