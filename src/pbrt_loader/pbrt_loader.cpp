#include "pbrt_loader.h"

#include <charconv>
#include <miniply.h>

#include "../color/spectral_data.h"

using namespace std::literals;

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
    sc.attribs = SceneAttribs::pbrt_defaults();

    load_screenwide_options(sc);
    load_scene_description(sc);
}

void
PbrtLoader::load_screenwide_options(Scene &sc) {
    while (true) {
        const auto directive = expect(LexemeType::String);

        if (directive.src == "Option") {
            spdlog::warn("'Option' ignored");
        } else if (directive.src == "Camera") {
            load_camera(sc);
        } else if (directive.src == "Sampler") {
            spdlog::warn("'Sampler' ignored");
            const auto _ = parse_param_list();
        } else if (directive.src == "ColorSpace") {
            spdlog::warn("'ColorSpace' ignored");
            const auto _ = parse_param_list();
        } else if (directive.src == "Film") {
            load_film(sc);
        } else if (directive.src == "PixelFilter") {
            spdlog::warn("'PixelFilter' ignored");
            const auto _ = parse_param_list();
        } else if (directive.src == "Integrator") {
            load_integrator(sc);
        } else if (directive.src == "Accelerator") {
            spdlog::warn("'Accelerator' ignored");
            const auto _ = parse_param_list();
        } else if (directive.src == "WorldBegin") {
            break;
        } else if (directive.src == "MakeNamedMedium") {
            spdlog::warn("'MakeNamedMedium' ignored");
            const auto _ = parse_param_list();
        } else if (directive.src == "MediumInterface") {
            spdlog::warn("'MediumInterface' ignored");
            const auto _ = parse_param_list();
        } else if (directive.src == "Identity") {
            load_identity();
        } else if (directive.src == "Translate") {
            load_translate();
        } else if (directive.src == "Scale") {
            load_scale();
        } else if (directive.src == "Rotate") {
            load_rotate();
        } else if (directive.src == "LookAt") {
            load_lookat();
        } else if (directive.src == "CoordinateSystem") {
            throw std::runtime_error("CoordinateSystem unimplemented");
        } else if (directive.src == "CoordSysTransform") {
            throw std::runtime_error("CoordSysTransform unimplemented");
        } else if (directive.src == "Transform") {
            load_transform();
        } else if (directive.src == "ConcatTransform") {
            load_concat_transform();
        } else {
            throw std::runtime_error(
                fmt::format("Unknown screenwide directive: '{}'", directive.src));
        }
    }

    assert(astates.empty());
    current_astate = AttributeState();
}

void
PbrtLoader::load_camera(Scene &sc) {
    // TODO: camera in mitsuba is right-handed but left-handed in PBRT
    auto params = parse_param_list();
    const auto &type = params.expect(ParamType::Simple);

    sc.attribs.camera.camera_to_world = current_astate.ctm.inverse();

    if (type.name == "perspective") {
        const auto fov = params.get_optional("fov", ValueType::Float);
        if (fov.has_value()) {
            sc.attribs.camera.fov = std::get<f32>(fov.value()->inner);
        }
    } else {
        spdlog::warn("Camera type '{}' unimplemented, using default", type.name);
    }

    params.warn_unused_params("Camera"sv);
}

void
PbrtLoader::load_film(Scene &sc) {
    auto params = parse_param_list();
    const auto &type = params.expect(ParamType::Simple);

    if (type.name != "rgb") {
        spdlog::warn("Film type is '{}', which is unimplemented, defaulting to RGB.",
                     type.name);
    }

    const auto resx = params.get_optional("xresolution", ValueType::Int);
    if (resx.has_value()) {
        sc.attribs.film.resx = std::get<i32>(resx.value()->inner);
    }

    const auto resy = params.get_optional("yresolution", ValueType::Int);
    if (resy.has_value()) {
        sc.attribs.film.resy = std::get<i32>(resy.value()->inner);
    }

    const auto filename = params.get_optional("filename", ValueType::String);
    if (filename.has_value()) {
        sc.attribs.film.filename = std::get<std::string>(filename.value()->inner);
    }

    params.warn_unused_params("Film"sv);
}

void
PbrtLoader::load_integrator(Scene &sc) {
    auto params = parse_param_list();

    const auto &maxdepth_p = params.get_optional("maxdepth", ValueType::Int);
    if (maxdepth_p.has_value()) {
        sc.attribs.max_depth = std::get<i32>(maxdepth_p.value()->inner);
    }

    params.warn_unused_params("Integrator");
}

void
PbrtLoader::load_identity() {
    current_astate.ctm = mat4::identity();
}

void
PbrtLoader::load_translate() {
    const auto x = parse_float();
    const auto y = parse_float();
    const auto z = parse_float();

    const auto trans = mat4::from_translate(x, y, z);
    current_astate.ctm = current_astate.ctm.compose(trans);
}

void
PbrtLoader::load_scale() {
    const auto x = parse_float();
    const auto y = parse_float();
    const auto z = parse_float();

    const auto trans = mat4::from_scale(x, y, z);
    current_astate.ctm = current_astate.ctm.compose(trans);
}

void
PbrtLoader::load_rotate() {
    const auto angle = to_rad(parse_float());
    const auto x = parse_float();
    const auto y = parse_float();
    const auto z = parse_float();

    // TODO: rotate will be wrong if negative axis is used
    mat4 trans;
    if (std::abs(x) == 1.0f && std::abs(y) == 0.0f && std::abs(z) == 0.0f) {
        trans = mat4::from_euler_x(angle);
    } else if (std::abs(x) == 0.0f && std::abs(y) == 1.0f && std::abs(z) == 0.0f) {
        trans = mat4::from_euler_y(angle);
    } else if (std::abs(x) == 0.0f && std::abs(y) == 0.0f && std::abs(z) == 1.0f) {
        trans = mat4::from_euler_z(angle);
    } else {
        throw std::runtime_error("Arbitrary euler angles rotate is unimplemented");
    }

    current_astate.ctm = current_astate.ctm.compose(trans);
}

void
PbrtLoader::load_lookat() {
    const auto eye_x = parse_float();
    const auto eye_y = parse_float();
    const auto eye_z = parse_float();
    const auto eye = vec3(eye_x, eye_y, eye_z);

    const auto look_x = parse_float();
    const auto look_y = parse_float();
    const auto look_z = parse_float();
    const auto look = vec3(look_x, look_y, look_z);

    const auto up_x = parse_float();
    const auto up_y = parse_float();
    const auto up_z = parse_float();
    const auto up = vec3(up_x, up_y, up_z);

    const auto trans = mat4::from_lookat(eye, look, up);
    current_astate.ctm = current_astate.ctm.compose(trans);
}

void
PbrtLoader::load_transform() {
    // Some files can have the numbers in brackets...
    auto p = parse_param("float", std::move(std::string("")));
    const auto &array = std::get<std::vector<f32>>(p.inner);

    if (array.size() != 16) {
        throw std::runtime_error("Transform has wrong amount of elements");
    }

    // clang-format off
    const auto trans = SquareMatrix4(
        array[0], array[1], array[2], array[3],
        array[4], array[5], array[6], array[7],
        array[8], array[9], array[10], array[11],
        array[12], array[13], array[14], array[15]
    );
    // clang-format on

    current_astate.ctm = trans;
}

void
PbrtLoader::load_concat_transform() {
    // Some files can have the numbers in brackets...
    auto p = parse_param("float", std::move(std::string("")));
    const auto &array = std::get<std::vector<f32>>(p.inner);

    if (array.size() != 16) {
        throw std::runtime_error("Transform has wrong amount of elements");
    }

    // clang-format off
    const auto trans = SquareMatrix4(
        array[0], array[1], array[2], array[3],
        array[4], array[5], array[6], array[7],
        array[8], array[9], array[10], array[11],
        array[12], array[13], array[14], array[15]
    );
    // clang-format on

    current_astate.ctm = current_astate.ctm.compose(trans);
}

void
PbrtLoader::load_scene_description(Scene &sc) {
    while (true) {
        if (lexer.peek().type == LexemeType::Eof) {
            break;
        }

        const auto directive = expect(LexemeType::String);

        if (directive.src == "MakeNamedMedium") {
            spdlog::warn("'MakeNamedMedium' ignored");
            const auto _ = parse_param_list();
        } else if (directive.src == "MediumInterface") {
            spdlog::warn("'MediumInterface' ignored");
            const auto _ = parse_param_list();
        } else if (directive.src == "Identity") {
            load_identity();
        } else if (directive.src == "Translate") {
            load_translate();
        } else if (directive.src == "Scale") {
            load_scale();
        } else if (directive.src == "Rotate") {
            load_rotate();
        } else if (directive.src == "LookAt") {
            load_lookat();
        } else if (directive.src == "CoordinateSystem") {
            throw std::runtime_error("CoordinateSystem unimplemented");
        } else if (directive.src == "CoordSysTransform") {
            throw std::runtime_error("CoordSysTransform unimplemented");
        } else if (directive.src == "Transform") {
            load_transform();
        } else if (directive.src == "ConcatTransform") {
            load_concat_transform();
        } else if (directive.src == "AttributeBegin") {
            attribute_begin();
        } else if (directive.src == "AttributeEnd") {
            attribute_end();
        } else if (directive.src == "Shape") {
            load_shape(sc);
        } else if (directive.src == "ObjectBegin") {
            // TODO: lexer context-sensitive fix...
            spdlog::warn("'ObjectBegin' ignored");
            const auto _ = parse_param_list();
        } else if (directive.src == "ObjectEnd") {
            spdlog::warn("'ObjectEnd' ignored");
            const auto _ = parse_param_list();
        } else if (directive.src == "ObjectInstance") {
            spdlog::warn("'ObjectInstance' ignored");
            const auto _ = parse_param_list();
        } else if (directive.src == "LightSource") {
            spdlog::warn("'LightSource' ignored");
            const auto _ = parse_param_list();
        } else if (directive.src == "AreaLightSource") {
            area_light_source(sc);
        } else if (directive.src == "ReverseOrientation") {
            current_astate.reverse_orientation = true;
        } else if (directive.src == "Material") {
            load_material(sc);
        } else if (directive.src == "MakeNamedMaterial") {
            load_make_named_material(sc);
        } else if (directive.src == "NamedMaterial") {
            load_named_material();
            const auto _ = parse_param_list();
        } else if (directive.src == "Texture") {
            load_texture(sc);
        } else {
            throw std::runtime_error(
                fmt::format("Unknown screenwide directive: '{}'", directive.src));
        }
    }
}

void
PbrtLoader::attribute_begin() {
    astates.push_back(current_astate);
}

void
PbrtLoader::attribute_end() {
    if (astates.empty()) {
        throw std::runtime_error("Spurious AttributeEnd");
    }

    current_astate = astates[astates.size() - 1];
    astates.pop_back();
}

void
PbrtLoader::load_shape(Scene &sc) {
    auto params = parse_param_list();

    const auto &type = params.next_param();
    if (type.name == "trianglemesh") {
        load_trianglemesh(sc, params);
    } else if (type.name == "plymesh") {
        load_plymesh(sc, params);
    } else if (type.name == "sphere") {
        spdlog::warn("Sphere shape ignored");
    } else {
        throw std::runtime_error(fmt::format("'{}' shape is unimplemented", type.name));
    }
}

// TODO: should I validate indices ?

void
PbrtLoader::load_trianglemesh(Scene &sc, ParamsList &params) const {
    std::vector<u32> indices{};
    std::vector<point3> pos{};
    std::vector<vec3> normals{};
    std::vector<vec2> uvs{};

    const auto &p_p = params.get_required("P", ValueType::Point3);
    const auto &pos_array = std::get<std::vector<point3>>(p_p.inner);
    pos = std::vector(pos_array);

    const auto indices_p = params.get_optional("indices", ValueType::Int);
    if (indices_p.has_value()) {
        auto &array = std::get<std::vector<i32>>(indices_p.value()->inner);
        indices = std::vector<u32>(array.begin(), array.end());
    }

    const auto n_p = params.get_optional("N", ValueType::Vector3);
    if (n_p.has_value()) {
        const auto &array = std::get<std::vector<vec3>>(n_p.value()->inner);
        normals = std::vector(array);
    }

    const auto uv_p = params.get_optional("uv", ValueType::Vector2);
    if (uv_p.has_value()) {
        const auto &array = std::get<std::vector<vec2>>(uv_p.value()->inner);
        uvs = std::vector(array);
    }

    if (pos.empty()) {
        throw std::runtime_error("'trianglemesh' Shape without positions");
    }

    // indices are required, unless exactly three vertices are specified.
    if (indices.empty()) {
        if (pos.size() == 3) {
            indices = std::vector{0u, 1u, 2u};
        } else {
            throw std::runtime_error("'trianglemesh' Shape without indices");
        }
    }

    if (current_astate.reverse_orientation) {
        for (auto &normal : normals) {
            normal = -normal;
        }
    }

    if (current_astate.reverse_orientation && normals.empty()) {
        // TODO: fix reverseorientation for this case
        spdlog::warn("ReverseOrientation without specified normals");
    }

    const auto mp = MeshParams{
        .indices = &indices,
        .pos = &pos,
        .normals = !normals.empty() ? &normals : nullptr,
        .uvs = !uvs.empty() ? &uvs : nullptr,
        .material_id = current_astate.material,
        .emitter = current_astate.emitter,
    };

    sc.add_mesh(mp);

    params.warn_unused_params("Shape trianglemesh"sv);
}

void
PbrtLoader::load_plymesh(Scene &sc, ParamsList &params) const {
    const auto &filename_p = params.get_required("filename", ValueType::String);
    const auto filename = std::get<std::string>(filename_p.inner);

    const auto filepath = absolute(file_directory).append(filename);
    miniply::PLYReader reader(filepath.c_str());
    if (!reader.valid()) {
        throw std::runtime_error(
            fmt::format("Can't read PLY file '{}'", filepath.string()));
    }

    std::vector<point3> pos{};
    std::vector<vec3> normals{};
    std::vector<vec2> uv{};
    std::vector<u32> indices{};

    u32 indexes[3];
    bool gotVerts = false;
    bool gotFaces = false;

    // Taken from the GitHub readme
    while (reader.has_element() && (!gotVerts || !gotFaces)) {
        if (reader.element_is(miniply::kPLYVertexElement) && reader.load_element() &&
            reader.find_pos(indexes)) {

            const auto numverts = reader.num_rows();
            pos.resize(numverts);

            reader.extract_properties(indexes, 3, miniply::PLYPropertyType::Float,
                                      pos.data());

            if (reader.find_texcoord(indexes)) {
                uv.resize(numverts);
                reader.extract_properties(indexes, 2, miniply::PLYPropertyType::Float,
                                          uv.data());
            }

            if (reader.find_normal(indexes)) {
                normals.resize(numverts);
                reader.extract_properties(indexes, 3, miniply::PLYPropertyType::Float,
                                          normals.data());
            }
            gotVerts = true;
        } else if (reader.element_is(miniply::kPLYFaceElement) && reader.load_element() &&
                   reader.find_indices(indexes)) {
            const bool polys = reader.requires_triangulation(indexes[0]);
            if (polys && !gotVerts) {
                throw std::runtime_error(fmt::format(
                    "PLY error in '{}': need vertex positions to triangulate faces.",
                    filename));
            }
            if (polys) {
                const auto num_indices = reader.num_triangles(indexes[0]) * 3;
                indices.resize(num_indices);

                reader.extract_triangles(indexes[0], &pos.data()->x, pos.size(),
                                         miniply::PLYPropertyType::Int, indices.data());
            } else {
                const auto num_indices = reader.num_rows() * 3;
                indices.resize(num_indices);

                reader.extract_list_property(indexes[0], miniply::PLYPropertyType::Int,
                                             indices.data());
            }
            gotFaces = true;
        }
        if (gotVerts && gotFaces) {
            break;
        }
        reader.next_element();
    }

    if (!gotVerts || !gotFaces || indices.empty() || pos.empty()) {
        throw std::runtime_error(fmt::format("PLY error in '{}'", filename));
    }

    if (current_astate.reverse_orientation && normals.empty()) {
        // TODO: fix reverseorientation for this case
        spdlog::warn("ReverseOrientation without specified normals");
    }

    if (current_astate.reverse_orientation) {
        for (auto &normal : normals) {
            normal = -normal;
        }
    }

    const auto mp = MeshParams{
        .indices = &indices,
        .pos = &pos,
        .normals = normals.empty() ? nullptr : &normals,
        .uvs = uv.empty() ? nullptr : &uv,
        .material_id = current_astate.material,
        .emitter = current_astate.emitter,
    };

    sc.add_mesh(mp);

    params.warn_unused_params("Shape plymesh"sv);
}

void
PbrtLoader::load_sphere(Scene &sc, ParamsList &params) const {
    auto radius = 1.f;

    const auto sphere_p = params.get_optional("radius", ValueType::Float);
    if (sphere_p.has_value()) {
        radius = std::get<f32>(sphere_p.value()->inner);
    }

    sc.add_sphere(SphereParams{.center = point3(0.f),
                               .radius = radius,
                               .material_id = current_astate.material,
                               .emitter = current_astate.emitter});

    params.warn_unused_params("Shape Sphere"sv);
}

// TODO: need to refactor the whole light-emitter nonsense... probably need to have the
//  emitter decoupled though, because texture emitter should definitely be shared...
void
PbrtLoader::area_light_source(Scene &sc) {
    auto params = parse_param_list();

    const auto &type = params.next_param();
    if (type.name != "diffuse") {
        throw std::runtime_error(
            fmt::format("Invalid area light source type", type.name));
    }

    auto twosided = false;
    auto radiance = Spectrum(
        RgbSpectrumIlluminant::make(tuple3(1.0f, 1.0f, 1.0f), current_astate.color_space),
        &sc.spectrum_allocator);
    auto scale = 1.f;

    const auto twosided_p = params.get_optional("twosided", ValueType::Bool);
    if (twosided_p.has_value()) {
        twosided = std::get<bool>(twosided_p.value()->inner);
    }

    const auto scale_p = params.get_optional("scale", ValueType::Float);
    if (scale_p.has_value()) {
        scale = std::get<f32>(scale_p.value()->inner);
    }

    const auto l_p = params.get_optional("L");
    if (l_p.has_value()) {
        const auto p = l_p.value();
        // TODO: will need some general spectrum-loader later
        if (p->value_type == ValueType::Rgb) {
            radiance =
                Spectrum(RgbSpectrumIlluminant::make(scale * std::get<tuple3>(p->inner),
                                                     current_astate.color_space),
                         &sc.spectrum_allocator);
        } else {
            throw std::runtime_error("AreaLight with radiance described other than "
                                     "in RGB is not implemented");
        }
    }

    Emitter emitter{radiance, twosided};

    // TODO: two-sided emitter is probably implemented badly... need to check area
    // calculations

    current_astate.emitter = emitter;

    params.warn_unused_params("AreaLightSource"sv);
}

void
PbrtLoader::load_material(Scene &sc) {
    auto params = parse_param_list();
    const auto &type_p = params.expect(ParamType::Simple);

    const auto mat = parse_material_description(sc, type_p.name, params);
    current_astate.material = sc.add_material(mat);

    params.warn_unused_params("Material"sv);
}

void
PbrtLoader::load_make_named_material(Scene &sc) {
    auto params = parse_param_list();

    const auto &name_p = params.expect(ParamType::Simple);

    const auto &type_p = params.expect(ParamType::Single);
    if (type_p.name != "type") {
        throw std::runtime_error("Wrong NamedMaterial type param");
    }

    const auto &type = std::get<std::string>(type_p.inner);
    const auto mat = parse_material_description(sc, type, params);

    const auto mat_id = sc.add_material(mat);
    materials.insert({name_p.name, mat_id});

    params.warn_unused_params("MakeNamedMaterial"sv);
}

Material
PbrtLoader::parse_material_description(Scene &sc, const std::string &type,
                                       ParamsList &params) {
    auto mat = Material::make_diffuse(TextureId{0});

    if (type == "coateddiffuse") {
        mat = parse_coateddiffuse_material(sc, params);
    } else if (type == "diffuse") {
        mat = parse_diffuse_material(sc, params);
    } else if (type == "dielectric") {
        mat = parse_dielectric_material(sc, params);
    } else if (type == "conductor") {
        mat = parse_conductor_material(sc, params);
    } else {
        spdlog::warn("Material '{}' is unimplemented, defaulting to diffuse", type);
    }

    mat.is_twosided = true;
    return mat;
}

RoughnessDescription
PbrtLoader::parse_material_roughness(Scene &sc, ParamsList &params) {
    const auto roughness_p = params.get_optional("roughness");
    if (roughness_p.has_value()) {
        const auto &roughness = *roughness_p.value();
        const auto tex = parse_inline_float_texture(roughness, sc);
        return RoughnessDescription{
            .type = RoughnessDescription::RoughnessType::Isotropic,
            .roughness = tex,
            .uroughness = TextureId{0},
            .vroughness = TextureId{0},
        };
    }

    const auto uroughness_opt = params.get_optional("uroughness");
    const auto vroughness_opt = params.get_optional("vroughness");

    if (uroughness_opt.has_value() && vroughness_opt.has_value()) {
        const auto &uroughness_p = *uroughness_opt.value();
        const auto &vroughness_p = *vroughness_opt.value();

        if (uroughness_p.value_type == ValueType::Float &&
            vroughness_p.value_type == ValueType::Float) {
            const auto uroughness = std::get<f32>(uroughness_p.inner);
            const auto vroughness = std::get<f32>(vroughness_p.inner);

            if (uroughness != vroughness) {
                spdlog::warn("Roughness Anisotropy isn't implemented yet");
            }

            const auto tex = sc.add_texture(Texture::make_constant_texture(uroughness));

            return RoughnessDescription{
                .type = RoughnessDescription::RoughnessType::Isotropic,
                .roughness = tex,
                .uroughness = TextureId{0},
                .vroughness = TextureId{0},
            };
        } else {
            spdlog::warn("Roughness Anisotropy isn't implemented yet");
            const auto tex =
                get_texture_id_or_default(sc, params, "uroughness", "roughness");
            return RoughnessDescription{
                .type = RoughnessDescription::RoughnessType::Isotropic,
                .roughness = tex,
                .uroughness = TextureId{0},
                .vroughness = TextureId{0},
            };
        }
    }

    const auto tex = get_texture_id_or_default(sc, params, "uroughness", "roughness");
    return RoughnessDescription{
        .type = RoughnessDescription::RoughnessType::Isotropic,
        .roughness = tex,
        .uroughness = TextureId{0},
        .vroughness = TextureId{0},
    };
}

Material
PbrtLoader::parse_coateddiffuse_material(Scene &sc, ParamsList &params) {
    const auto ext_ior = Spectrum(AIR_ETA);
    // TODO: get coateddiffuse IOR
    const auto int_ior = Spectrum(POLYPROPYLENE_ETA);
    const auto reflectance =
        get_texture_id_or_default(sc, params, "reflectance", "reflectance");

    const auto roughness = parse_material_roughness(sc, params);

    return Material::make_rough_plastic(roughness.roughness, ext_ior, int_ior,
                                        reflectance, sc.material_allocator);
}

Material
PbrtLoader::parse_diffuse_material(Scene &sc, ParamsList &params) {
    const auto texture =
        get_texture_id_or_default(sc, params, "reflectance", "reflectance");
    return Material::make_diffuse(texture);
}

Material
PbrtLoader::parse_dielectric_material(Scene &sc, ParamsList &params) {
    // TODO: dielectric rough material not implemented
    // TODO: move this inside the scene spectra...
    // const auto ext_ior = Spectrum(AIR_ETA);
    const auto ext_ior = Spectrum(ConstantSpectrum::make(1.f));
    const auto trans = Spectrum(ConstantSpectrum::make(1.f));
    const auto int_ior = get_texture_id_or_default(sc, params, "eta", "eta-dielectric");

    return Material::make_dielectric(ext_ior, int_ior, trans, sc.material_allocator);
}

Material
PbrtLoader::parse_conductor_material(Scene &sc, ParamsList &params) {
    const auto eta = get_texture_id_or_default(sc, params, "eta", "eta-conductor");
    const auto k = get_texture_id_or_default(sc, params, "k", "k-conductor");
    const auto roughness = parse_material_roughness(sc, params);

    return Material::make_rough_conductor(roughness.roughness, eta, k,
                                          sc.material_allocator);
}

TextureId
PbrtLoader::get_texture_id_or_default(Scene &sc, ParamsList &params,
                                      const std::string &name,
                                      const std::string &default_tex) {
    const auto opt_p = params.get_optional(name);

    if (!opt_p.has_value()) {
        return sc.builtin_textures.at(default_tex);
    }

    const auto p = opt_p.value();
    if (p->value_type == ValueType::Texture) {
        const auto &tex_name = std::get<std::string>(p->inner);
        return textures.at(tex_name);
    }

    if (p->name == "eta") {
        return parse_inline_spectrum_texture(*p, sc);
    } else if (p->name == "k") {
        return parse_inline_spectrum_texture(*p, sc);
    } else if (p->name == "reflectance") {
        return parse_inline_spectrum_texture(*p, sc);
    } else {
        throw std::runtime_error(fmt::format("Unimplemented texture: '{}'", p->name));
    }
}

// TODO: probably should validate here, that the texture has correct params

TextureId
PbrtLoader::parse_inline_spectrum_texture(const Param &param, Scene &sc) {
    if (param.value_type == ValueType::Rgb) {
        const auto spectrum = RgbSpectrum::make(std::get<tuple3>(param.inner));
        return sc.add_texture(Texture::make_constant_texture(spectrum));
    } else if (param.value_type == ValueType::String) {
        return sc.builtin_textures.at(std::get<std::string>(param.inner));
    } else if (param.value_type == ValueType::Float) {
        return sc.add_texture(Texture::make_constant_texture(std::get<f32>(param.inner)));
    } else {
        // TODO: implement other spectrum types
        spdlog::warn("Spectrum texture '{}' unimplemented, getting default", param.name);
        return sc.add_texture(
            Texture::make_constant_texture(RgbSpectrum::make(tuple3(0.8, 0.5, 0.3))));
    }
}

TextureId
PbrtLoader::parse_inline_float_texture(const Param &param, Scene &sc) {
    if (param.value_type == ValueType::Float) {
        const auto fl = std::get<f32>(param.inner);
        return sc.add_texture(Texture::make_constant_texture(fl));
    } else if (param.value_type == ValueType::Texture) {
        const auto texture_name = std::get<std::string>(param.inner);
        return textures.at(texture_name);
    } else {
        // TODO: implement other spectrum types
        spdlog::warn("Float texture '{}' unimplemented, getting default", param.name);
        return sc.add_texture(Texture::make_constant_texture(1.f));
    }
}

void
PbrtLoader::load_named_material() {
    auto params = parse_param_list();
    const auto &p = params.next_param();
    current_astate.material = materials.at(p.name);
}

void
PbrtLoader::load_texture(Scene &sc) {
    auto params = parse_param_list();

    const auto &name = params.expect(ParamType::Simple).name;
    const auto &type = params.expect(ParamType::Simple).name;
    if (type != "spectrum" && type != "float") {
        throw std::runtime_error(fmt::format("Invalid texture type: '{}'", type));
    }

    const auto &tex_class = params.expect(ParamType::Simple).name;

    Texture tex;

    if (tex_class == "imagemap") {
        const auto &filename_p = params.get_required("filename", ValueType::String);
        const auto filename = std::get<std::string>(filename_p.inner);

        const auto is_rgb = type == "spectrum";
        const auto path = absolute(file_directory).append(filename);
        tex = Texture::make_image_texture(path, is_rgb);
    } else if (tex_class == "scale") {
        const auto &tex_p = params.get_required("tex", ValueType::Texture);
        const auto tex_name = std::get<std::string>(tex_p.inner);

        const auto id = textures.at(tex_name);
        textures.insert({name, id});
    } else {
        throw std::runtime_error(
            fmt::format("Unimplemented texture class: '{}'", tex_class));
    }

    const auto id = sc.add_texture(tex);
    textures.insert({name, id});

    params.warn_unused_params("Texture");
}

ParamsList
PbrtLoader::parse_param_list() {
    ParamsList params{};

    // The opening quotes of a parameter
    while (lexer.peek().type == LexemeType::Quotes) {
        lexer.next();

        if (lexer.peek().type == LexemeType::Quotes) {
            // Handle empty string param (used in MediumInterface...)
            lexer.next();

            auto param = Param(std::move(std::string{""}));
            params.add(std::move(param));

            continue;
        }

        auto type_or_param = expect(LexemeType::String);

        if (lexer.peek().type == LexemeType::Quotes) {
            // A simple parameter
            lexer.next();

            auto param = Param(std::move(type_or_param.src));
            params.add(std::move(param));

            continue;
        }

        // type_or_param is a type
        auto name = expect(LexemeType::String);
        expect(LexemeType::Quotes);

        auto param = parse_param(type_or_param.src, std::move(name.src));
        params.add(std::move(param));
    }

    return params;
}

Param
PbrtLoader::parse_param(const std::string_view &type, std::string &&name) {
    auto has_brackets = false;
    if (lexer.peek().type == LexemeType::OpenBracket) {
        has_brackets = true;
        lexer.next();
    }

    Param param{};

    if (type == "integer") {
        param = parse_value_list<i32>([this] { return parse_int(); }, has_brackets,
                                      std::move(name));
    } else if (type == "float") {
        param = parse_value_list<f32>([this] { return parse_float(); }, has_brackets,
                                      std::move(name));
    } else if (type == "point2") {
        param = parse_value_list<vec2>([this] { return parse_vec2(); }, has_brackets,
                                       std::move(name));
    } else if (type == "vector2") {
        param = parse_value_list<vec2>([this] { return parse_vec2(); }, has_brackets,
                                       std::move(name));
    } else if (type == "point3") {
        param = parse_value_list<point3>([this] { return parse_point3(); }, has_brackets,
                                         std::move(name));
    } else if (type == "vector3") {
        param = parse_value_list<vec3>([this] { return parse_vec3(); }, has_brackets,
                                       std::move(name));
    } else if (type == "normal" || type == "normal3") {
        param = parse_value_list<vec3>([this] { return parse_normal(); }, has_brackets,
                                       std::move(name));
    } else if (type == "spectrum") {
        param = parse_spectrum_param(std::move(name));
    } else if (type == "rgb") {
        const auto rgb = parse_rgb();
        param = Param(std::move(name), rgb);
    } else if (type == "blackbody") {
        throw std::runtime_error("blackbody param unimplemented");
    } else if (type == "bool") {
        const auto _bool = parse_bool();
        param = Param(std::move(name), _bool);
    } else if (type == "string") {
        param = parse_value_list<std::string>([this] { return parse_quoted_string(); },
                                              has_brackets, std::move(name));
    } else if (type == "texture") {
        auto str = parse_quoted_string();
        param = Param(std::move(name), TextureValue{.str = std::move(str)});
    } else {
        throw std::runtime_error(fmt::format("Unknown param type: '{}'", type));
    }

    if (has_brackets) {
        expect(LexemeType::CloseBracket);
    }

    return param;
}

Param
PbrtLoader::parse_spectrum_param(std::string &&name) {
    auto has_brackets = false;
    if (lexer.peek().type == LexemeType::OpenBracket) {
        lexer.next();
        has_brackets = true;
    }

    Param param{};

    const auto first_lexeme = lexer.peek();
    if (first_lexeme.type == LexemeType::Quotes) {
        lexer.next();
        auto spectrum_name = expect(LexemeType::String);
        param = Param(std::move(name), std::move(spectrum_name.src));

        expect(LexemeType::Quotes);
    } else if (first_lexeme.type == LexemeType::Num) {
        std::vector<f32> vals{};

        while (lexer.peek().type == LexemeType::Num) {
            const auto val = parse_float();
            vals.push_back(val);
        }

        param = Param(std::move(name), std::move(vals), ValueType::Spectrum);
    }

    if (has_brackets) {
        expect(LexemeType::CloseBracket);
    }

    return param;
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

vec3
PbrtLoader::parse_normal() {
    const auto x = parse_float();
    const auto y = parse_float();
    const auto z = parse_float();
    return vec3(x, y, z);
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
    if (lexer.peek().type == LexemeType::Quotes) {
        lexer.next();
        return std::string{""};
    }

    const auto lex = expect(LexemeType::String);
    expect(LexemeType::Quotes);

    return lex.src;
}

Lexeme
PbrtLoader::expect(const LexemeType lt) {
    const auto lex = lexer.next();
    if (lex.type != lt) {
        throw std::runtime_error(
            fmt::format("wrong lexeme type, lexeme: '{}' on line '{}'", lex.src,
                        lexer.newline_counter));
    } else {
        return lex;
    }
}
