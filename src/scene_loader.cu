#include "scene_loader.h"

#include <array>
#include <ranges>
#include <utility>

#include "color/spectral_data.h"
#include "color/spectrum.h"
#include "math/vecmath.h"
#include <fmt/core.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using str = std::string_view;

void
SceneLoader::load_scene(Scene *sc) {
    auto scene = doc.child("scene");

    load_materials(scene, sc);
    load_shapes(sc, scene);

    auto envmap_node = scene.child("emitter");
    if (envmap_node) {
        std::string filename = envmap_node.child("string").attribute("value").as_string();
        auto file_path = scene_base_path + "/" + filename;

        auto to_world_transform = mat4::identity();
        auto transform_node = envmap_node.child("transform");
        to_world_transform = parse_transform(transform_node);

        auto envmap = Envmap(file_path, to_world_transform);
        sc->set_envmap(std::move(envmap));
    }
}

void
SceneLoader::load_shapes(Scene *sc, const pugi::xml_node &scene) {
    for (pugi::xml_node shape : scene.children("shape")) {
        str type = shape.attribute("type").as_string();

        u32 mat_id;

        auto ref_node = shape.child("ref");
        auto bsdf_node = shape.child("bsdf");
        if (ref_node) {
            std::string bsdf_id = ref_node.attribute("id").as_string();
            mat_id = materials.at(bsdf_id);
        } else if (bsdf_node) {
            auto [mat, _] = load_material(sc, bsdf_node);
            mat_id = sc->add_material(std::move(mat));
        } else {
            throw std::runtime_error("Shape has no material");
        }

        auto transform_node = shape.child("transform");
        mat4 transform = mat4::identity();
        if (transform_node) {
            transform = parse_transform(transform_node);
        }

        auto emitter_node = shape.child("emitter");
        COption<Emitter> emitter = {};
        if (emitter_node) {
            emitter = load_emitter(emitter_node, sc);
        }

        if (type == "rectangle") {
            load_rectangle(shape, mat_id, transform, emitter, sc);
        } else if (type == "cube") {
            load_cube(shape, mat_id, transform, emitter, sc);
        } else if (type == "obj") {
            load_obj(shape, mat_id, transform, emitter, sc);
        } else if (type == "sphere") {
            load_sphere(shape, mat_id, transform, emitter, sc);
        } else {
            throw std::runtime_error(fmt::format("Unknown shape type: {}", type));
        }
    }
}

void
SceneLoader::load_materials(pugi::xml_node scene, Scene *sc) {
    auto bsdfs = scene.children("bsdf");

    for (auto bsdf : bsdfs) {
        auto [mat, id] = load_material(sc, bsdf);

        u32 mat_id = sc->add_material(std::move(mat));
        materials.insert({id, mat_id});
    }
}

std::tuple<Material, std::string>
SceneLoader::load_material(Scene *sc, pugi::xml_node &bsdf) {
    str type = bsdf.attribute("type").as_string();
    std::string id = bsdf.attribute("id").as_string();
    bool is_twosided = false;

    if (type == "twosided" || type == "bumpmap") {
        bsdf = bsdf.child("bsdf");
        type = bsdf.attribute("type").as_string();
        is_twosided = true;

        // For bumpmaps, the id is in the nested bsdf... ??
        if (id.empty()) {
            id = bsdf.attribute("id").as_string();
        }
    }

    // Default material - 50% reflectance
    Material mat = Material::make_diffuse(RgbSpectrum::make(tuple3(0.5f)));

    if (type == "diffuse") {
        auto reflectance_node = bsdf.find_child([](pugi::xml_node node) {
            return node.find_attribute([](pugi::xml_attribute attr) {
                return str(attr.name()) == "name" &&
                       str(attr.as_string()) == "reflectance";
            });
        });

        if (str(reflectance_node.name()) == "texture") {
            auto filename_node = reflectance_node.find_child([](pugi::xml_node node) {
                return node.find_attribute([](pugi::xml_attribute attr) {
                    return str(attr.name()) == "name" &&
                           str(attr.as_string()) == "filename";
                });
            });

            auto file_path = filename_node.attribute("value").as_string();
            auto texture = Texture::make(this->scene_base_path + "/" + file_path, true);

            u32 tex_id = sc->add_texture(std::move(texture));
            mat = Material::make_diffuse(tex_id);
        } else {
            tuple3 rgb = parse_tuple3(reflectance_node.attribute("value").as_string());
            mat = Material::make_diffuse(RgbSpectrum::make(rgb));
        }
    } else if (type == "conductor") {
        mat = Material::make_conductor();
    } else if (type == "dielectric") {
        auto ext_ior = AIR_ETA;
        auto int_ior = GLASS_BK7_ETA;
        spdlog::info("Mitsuba dielectric IORs are ignored for now");

        auto transmittance_node = bsdf.find_child([](pugi::xml_node node) {
            return node.find_attribute([](pugi::xml_attribute attr) {
                return str(attr.name()) == "name" &&
                       str(attr.as_string()) == "specular_transmittance";
            });
        });

        Spectrum specular_transmittance = Spectrum(ConstantSpectrum::make(1.f));
        if (transmittance_node) {
            tuple3 rgb = parse_tuple3(transmittance_node.attribute("value").as_string());
            specular_transmittance = Spectrum(RgbSpectrum::make(rgb));
        }

        mat = Material::make_dielectric(Spectrum(ext_ior), Spectrum(int_ior),
                                        specular_transmittance);
    } else {
        spdlog::warn("Unknown BSDF type: {}, defaulting to diffuse", type);
    }

    mat.is_twosided = is_twosided;

    return {mat, id};
}

mat4
SceneLoader::parse_transform(pugi::xml_node transform_node) {
    mat4 cur_transform = mat4::identity();

    for (auto child_node : transform_node.children()) {
        str name = child_node.name();

        if (name == "matrix") {
            str matrix_str = child_node.attribute("value").as_string();

            auto floats =
                std::views::transform(std::views::split(matrix_str, ' '), [](auto v) {
                    auto c = v | std::views::common;
                    return std::stof(std::string(c.begin(), c.end()));
                });

            CArray<f32, 16> mat{};
            int i = 0;
            for (f32 f : floats) {
                if (i > 15) {
                    throw std::runtime_error("Wrong matrix element count");
                }

                mat[i] = f;

                i++;
            }

            // matrices are stored in column-majorm, but Mitsuba's format is row-major...
            cur_transform = cur_transform.compose(mat4::from_elements(mat).transpose());
        } else if (name == "rotate") {
            auto angle_attr = child_node.attribute("angle");
            if (angle_attr) {
                f32 angle = to_rad(angle_attr.as_float());

                mat4 trans = mat4::identity();
                if (child_node.attribute("y")) {
                    trans = mat4::from_euler_y(angle);
                } else if (child_node.attribute("x")) {
                    trans = mat4::from_euler_x(angle);
                } else if (child_node.attribute("z")) {
                    trans = mat4::from_euler_z(angle);
                }
                cur_transform = cur_transform.compose(trans);
            } else {
                throw std::runtime_error("Rotate along arbitrary axis not implemented");
            }
        } else {
            throw std::runtime_error(fmt::format("Unknown transform type: {}", name));
        }
    }

    return cur_transform;
}

tuple3
SceneLoader::parse_tuple3(const std::string &str) {
    auto floats = std::views::transform(std::views::split(str, ' '), [](auto v) {
        auto c = v | std::views::common;
        return std::stof(std::string(c.begin(), c.end()));
    });

    tuple3 rgb(0.);

    int i = 0;
    for (f32 f : floats) {
        rgb[i] = f;
        if (i > 2) {
            throw std::runtime_error("Wrong tuple3 element count");
        }

        i++;
    }

    return rgb;
}

Emitter
SceneLoader::load_emitter(pugi::xml_node emitter_node, Scene *sc) {
    str type = emitter_node.attribute("type").as_string();
    if (type != "area") {
        throw std::runtime_error(fmt::format("Unknown emitter type: {}", type));
    }

    auto rgb_node = emitter_node.child("rgb");
    if (!rgb_node) {
        throw std::runtime_error("Emitter doesn't have rgb");
    }

    tuple3 emittance_rgb = parse_tuple3(rgb_node.attribute("value").as_string());
    auto emittance = RgbSpectrumIlluminant::make(emittance_rgb, ColorSpace::sRGB);

    return Emitter(emittance);
}

void
SceneLoader::load_rectangle(pugi::xml_node shape, u32 mat_id, const mat4 &transform,
                            COption<Emitter> emitter, Scene *sc) {
    // clang-format off
    UmVector<point3> pos = {
        point3(-1.,  -1., 0.),
        point3( 1.,  -1., 0.),
        point3( 1.,   1., 0.),
        point3(-1.,   1., 0.)
    };

    /*
     * 3 -- 2
     * |    |
     * 0 -- 1
     * */
    UmVector<u32> indices{
        0, 1, 2, 0, 2, 3,
    };
    // clang-format on

    for (int i = 0; i < pos.size(); i++) {
        pos[i] = transform.transform_point(pos[i]);
    }

    MeshParams mp = {
        .indices = &indices,
        .pos = &pos,
        .material_id = mat_id,
        .emitter = emitter,
    };

    sc->add_mesh(mp);
}

void
SceneLoader::load_cube(pugi::xml_node shape, u32 mat_id, const mat4 &transform,
                       COption<Emitter> emitter, Scene *sc) {
    // clang-format off
    UmVector<point3> pos = {
        point3(-1.,  -1.,  1.),
        point3( 1.,  -1.,  1.),
        point3( 1.,   1.,  1.),
        point3(-1.,   1.,  1.),

        point3(-1.,  -1., -1.),
        point3( 1.,  -1., -1.),
        point3( 1.,   1., -1.),
        point3(-1.,   1., -1.),
    };

    /* Front face     back face
     * 3 -- 2         7 -- 6
     * |    |         |    |
     * 0 -- 1         4 -- 5
     * */
    UmVector<u32> indices{
        0, 1, 2, 0, 2, 3,
        1, 5, 6, 1, 6, 2,
        3, 2, 6, 3, 6, 7,
        4, 0, 3, 4, 3, 7,
        5, 4, 7, 5, 7, 6,
        4, 5, 1, 4, 1, 0,
    };
    // clang-format on

    for (int i = 0; i < pos.size(); i++) {
        pos[i] = transform.transform_point(pos[i]);
    }

    MeshParams mp = {
        .indices = &indices,
        .pos = &pos,
        .material_id = mat_id,
        .emitter = emitter,
    };

    sc->add_mesh(mp);
}

void
SceneLoader::load_sphere(pugi::xml_node node, u32 mat_id, mat4 transform,
                         COption<Emitter> emitter, Scene *sc) {
    auto radius_node = node.find_child([](pugi::xml_node node) {
        return node.find_attribute([](pugi::xml_attribute attr) {
            return str(attr.name()) == "name" && str(attr.as_string()) == "radius";
        });
    });

    f32 radius = radius_node.attribute("value").as_float();

    auto center_node = node.find_child([](pugi::xml_node node) {
        return node.find_attribute([](pugi::xml_attribute attr) {
            return str(attr.name()) == "name" && str(attr.as_string()) == "center";
        });
    });

    f32 x = center_node.attribute("x").as_float();
    f32 y = center_node.attribute("y").as_float();
    f32 z = center_node.attribute("z").as_float();

    point3 center = point3(x, y, z);
    auto sphere = SphereParams{center, radius, mat_id, emitter};
    sc->add_sphere(sphere);
}

void
SceneLoader::load_obj(pugi::xml_node shape_node, u32 mat_id, const mat4 &transform,
                      COption<Emitter> emitter, Scene *sc) {
    std::string filename = shape_node.child("string").attribute("value").as_string();
    auto file_path = scene_base_path + "/" + filename;

    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./";
    reader_config.triangulate = true;

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(file_path, reader_config)) {
        if (!reader.Error().empty()) {
            throw std::runtime_error(
                fmt::format("Error reading obj file: '{}'", reader.Error()));
        } else {
            throw std::runtime_error("Error reading obj file");
        }
    }

    if (!reader.Warning().empty()) {
        spdlog::warn("Warning when reading obj file");
    }

    bool face_normals = false;
    auto face_normals_node = shape_node.find_child([](pugi::xml_node node) {
        return node.find_attribute([](pugi::xml_attribute attr) {
            return str(attr.name()) == "name" && str(attr.value()) == "face_normals";
        });
    });
    if (face_normals_node) {
        face_normals = face_normals_node.attribute("value").as_bool();
    }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();

    UmVector<point3> pos{};
    UmVector<vec3> normals{};
    UmVector<vec2> uvs{};
    UmVector<u32> indices{};

    if (shapes.size() != 1) {
        throw std::runtime_error(
            fmt::format("OBJ doesn't have exactly one shape: {}", file_path));
    }
    auto shape = &shapes[0];

    // Wavefront OBJ format is so cursed....
    // Load indices
    for (size_t f = 0; f < shape->mesh.num_face_vertices.size(); f++) {
        auto fv = size_t(shape->mesh.num_face_vertices[f]);
        if (fv != 3) {
            // Triangulation should take care of this...
            throw std::runtime_error(
                fmt::format("OBJ file has non-triangle faces: {}", file_path));
        }

        int i0 = shape->mesh.indices[3 * f].vertex_index;
        int i1 = shape->mesh.indices[3 * f + 1].vertex_index;
        int i2 = shape->mesh.indices[3 * f + 2].vertex_index;

        for (int i = 0; i < 3; i++) {
            auto ind = shape->mesh.indices[3 * f + i];
            bool matching_indices = (ind.vertex_index == ind.normal_index &&
                                     ind.vertex_index == ind.texcoord_index);

            if (!matching_indices) {
                throw std::runtime_error(
                    fmt::format("OBJ file doesn't have correct indices: {}", file_path));
            }
        }

        indices.push(i0);
        indices.push(i1);
        indices.push(i2);
    }

    size_t pos_size = attrib.vertices.size() / 3;
    size_t normals_size = attrib.normals.size() / 3;
    size_t uvs_size = attrib.texcoords.size() / 2;

    if (pos_size < 3) {
        throw std::runtime_error(
            fmt::format("OBJ file doesn't have enough vertices: {}", file_path));
    }

    if (pos_size != normals_size || pos_size != uvs_size) {
        throw std::runtime_error(fmt::format(
            "OBJ file has non-matching number of vertex attributes: {}", file_path));
    }

    // Copy vertices
    for (int v = 0; v < pos_size; v++) {
        tinyobj::real_t x = attrib.vertices[3 * v];
        tinyobj::real_t y = attrib.vertices[3 * v + 1];
        tinyobj::real_t z = attrib.vertices[3 * v + 2];

        point3 vert_pos = point3(x, y, z);
        pos.push(vert_pos);

        if (!face_normals) {
            tinyobj::real_t nx = attrib.normals[3 * v];
            tinyobj::real_t ny = attrib.normals[3 * v + 1];
            tinyobj::real_t nz = attrib.normals[3 * v + 2];

            vec3 vert_normal = vec3(nx, ny, nz);
            normals.push(vert_normal);
        }

        tinyobj::real_t tx = attrib.texcoords[2 * v];
        tinyobj::real_t ty = attrib.texcoords[2 * v + 1];

        vec2 uv = vec2(tx, ty);
        uvs.push(uv);
    }

    for (int i = 0; i < pos.size(); i++) {
        pos[i] = transform.transform_point(pos[i]);
    }

    auto inv_trans = transform.transpose().inverse();
    for (int i = 0; i < normals.size(); i++) {
        normals[i] = inv_trans.transform_vec(normals[i]);
    }

    MeshParams mp = {
        .indices = &indices,
        .pos = &pos,
        .normals = (face_normals) ? nullptr : &normals,
        .uvs = &uvs,
        .material_id = mat_id,
        .emitter = std::move(emitter),
    };

    sc->add_mesh(mp);
}

std::optional<SceneAttribs>
SceneLoader::load_scene_attribs() {
    SceneAttribs attribs;
    auto scene = doc.child("scene");

    for (pugi::xml_node def : scene.children("default")) {
        str name = def.attribute("name").as_string();
        if (name == "resx") {
            attribs.resx = def.attribute("value").as_int();
        } else if (name == "resy") {
            attribs.resy = def.attribute("value").as_int();
        }
    }

    auto sensor = scene.child("sensor");
    for (pugi::xml_node f : sensor.children("float")) {
        str name = f.attribute("name").as_string();
        if (name == "fov") {
            attribs.fov = f.attribute("value").as_float();
        }
    }

    auto transform = sensor.child("transform");
    if (transform) {
        attribs.camera_to_world = parse_transform(transform);
    }

    return std::make_optional<SceneAttribs>(attribs);
}
