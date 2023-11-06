#include "scene_loader.h"

#include <array>
#include <ranges>

#include <fmt/core.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

void SceneLoader::load_scene(RenderContext *rc) {
    auto scene = doc.child("scene");

    load_materials(scene, rc);
    load_shapes(rc, scene);

    auto envmap_node = scene.child("emitter");
    if (envmap_node) {
        std::string filename = envmap_node.child("string").attribute("value").as_string();
        auto file_path = scene_base_path + "/" + filename;
        auto envmap = Envmap(file_path);
        rc->set_envmap(std::move(envmap));
    }
}

void SceneLoader::load_shapes(RenderContext *rc, const pugi::xml_node &scene) {
    for (pugi::xml_node shape : scene.children("shape")) {
        std::string type = shape.attribute("type").as_string();

        auto ref_node = shape.child("ref");
        if (!ref_node) {
            // TODO: the bsdf can be specified inline
            spdlog::error("Shape has no ref node");
            throw;
        }

        std::string bsdf_id = ref_node.attribute("id").as_string();
        u32 mat_id = materials.at(bsdf_id);

        auto transform_node = shape.child("transform");
        mat4 transform = mat4(1.);
        if (transform_node) {
            transform = parse_transform(transform_node);
        }

        auto emitter_node = shape.child("emitter");
        cuda::std::optional<u32> light_id = cuda::std::nullopt;
        if (emitter_node) {
            light_id = {load_emitter(emitter_node, rc)};
        }

        if (type == "rectangle") {
            load_rectangle(shape, mat_id, transform, light_id, rc);
        } else if (type == "cube") {
            load_cube(shape, mat_id, transform, light_id, rc);
        } else if (type == "obj") {
            load_obj(shape, mat_id, transform, light_id, rc);
        } else if (type == "sphere") {
            spdlog::debug("Ingoring sphere shape");
        } else {
            spdlog::error("Unknown shape type: {}", type);
            throw;
        }
    }
}

void SceneLoader::load_materials(pugi::xml_node scene, RenderContext *rc) {
    auto bsdfs = scene.children("bsdf");

    for (auto bsdf : bsdfs) {
        std::string id = bsdf.attribute("id").as_string();
        std::string type = bsdf.attribute("type").as_string();

        if (type == "twosided") {
            bsdf = bsdf.child("bsdf");
            type = bsdf.attribute("type").as_string();
        }

        Material mat = Material(vec3(0.5, 0.5, 0.5));

        if (type == "diffuse") {
            auto reflectance_node = bsdf.find_child([](pugi::xml_node node) {
                return node.find_attribute([](pugi::xml_attribute attr) {
                    return strcmp(attr.name(), "name") == 0 &&
                           (strcmp(attr.as_string(), "reflectance") == 0);
                });
            });

            if (strcmp(reflectance_node.name(), "texture") == 0) {
                auto filename_node = reflectance_node.find_child([](pugi::xml_node node) {
                    return node.find_attribute([](pugi::xml_attribute attr) {
                        return strcmp(attr.name(), "name") == 0 &&
                               strcmp(attr.as_string(), "filename") == 0;
                    });
                });

                auto file_path = filename_node.attribute("value").as_string();
                auto texture = Texture(scene_base_path + "/" + file_path);

                u32 tex_id = rc->add_texture(std::move(texture));
                mat = Material(tex_id);
            } else {
                vec3 rgb = parse_rgb(reflectance_node.attribute("value").as_string());
                mat = Material(rgb);
            }

        } else {
            spdlog::debug("Unknown BSDF type: {}, defaulting to diffuse", type);
        }

        u32 mat_id = rc->add_material(std::move(mat));
        materials.insert({id, mat_id});
    }
}

mat4 SceneLoader::parse_transform(pugi::xml_node transform_node) {
    auto matrix = transform_node.child("matrix");
    if (matrix) {
        std::string matrix_str = matrix.attribute("value").as_string();
        auto floats =
            std::views::transform(std::views::split(matrix_str, ' '), [](auto v) {
                auto c = v | std::views::common;
                return std::stof(std::string(c.begin(), c.end()));
            });

        std::array<f32, 16> mat{};
        int i = 0;
        for (f32 f : floats) {
            if (i > 15) {
                spdlog::error("Wrong matrix element count");
                throw;
            }

            mat[i] = f;

            i++;
        }

        // GLM stores matrices in column-majorm, but Mitsuba's format is row-major...
        return glm::transpose(glm::make_mat4(mat.data()));
    } else {
        spdlog::error("Unknown transform type");
        throw;
    }
}

vec3 SceneLoader::parse_rgb(const std::string &str) {
    auto floats = std::views::transform(std::views::split(str, ' '), [](auto v) {
        auto c = v | std::views::common;
        return std::stof(std::string(c.begin(), c.end()));
    });

    vec3 rgb(0.);

    int i = 0;
    for (f32 f : floats) {
        rgb[i] = f;
        if (i > 2) {
            spdlog::error("Wrong rgb element count");
            throw;
        }

        i++;
    }

    return rgb;
}

u32 SceneLoader::load_emitter(pugi::xml_node emitter_node, RenderContext *rc) {
    std::string type = emitter_node.attribute("type").as_string();
    if (type != "area") {
        spdlog::error("Unknown emitter type");
        throw;
    }

    auto rgb_node = emitter_node.child("rgb");
    if (!rgb_node) {
        spdlog::error("Emitter doesn't have rgb");
        throw;
    }

    vec3 emittance = parse_rgb(rgb_node.attribute("value").as_string());

    u32 light_id = rc->add_light(Light(emittance));
    return light_id;
}

void SceneLoader::load_rectangle(pugi::xml_node shape, u32 mat_id, const mat4 &transform,
                                 cuda::std::optional<u32> light_id, RenderContext *rc) {
    // clang-format off
    SharedVector<vec3> pos = {vec3(-1.,  -1., 0.),
                               vec3( 1.,  -1., 0.),
                               vec3( 1.,   1., 0.),
                               vec3(-1.,   1., 0.)
    };

    /*
     * 3 -- 2
     * |    |
     * 0 -- 1
     * */
    SharedVector<u32> indices{
        0, 1, 2, 0, 2, 3,
    };
    // clang-format on

    for (int i = 0; i < pos.len(); i++) {
        pos[i] = transform * vec4(pos[i], 1.);
    }

    MeshParams mp = {
        .indices = &indices,
        .pos = &pos,
        .material_id = mat_id,
        .light_id = light_id,
    };

    rc->add_mesh(mp);
}

void SceneLoader::load_cube(pugi::xml_node shape, u32 mat_id, const mat4 &transform,
                            cuda::std::optional<u32> light_id, RenderContext *rc) {
    // clang-format off
    SharedVector<vec3> pos = {vec3(-1.,  -1., -1.),
                              vec3( 1.,  -1., -1.),
                              vec3( 1.,   1., -1.),
                              vec3(-1.,   1., -1.),

                              vec3(-1.,  -1.,  1.),
                              vec3( 1.,  -1.,  1.),
                              vec3( 1.,   1.,  1.),
                              vec3(-1.,   1.,  1.),
    };

    /* Front face     back face
     * 3 -- 2         7 -- 6
     * |    |         |    |
     * 0 -- 1         4 -- 5
     * */
    SharedVector<u32> indices{
        0, 1, 2, 0, 2, 3,
        1, 5, 6, 1, 6, 2,
        3, 2, 6, 3, 6, 7,
        4, 0, 3, 4, 3, 7,
        5, 4, 7, 5, 7, 6,
        4, 5, 1, 4, 1, 0,
    };
    // clang-format on

    for (int i = 0; i < pos.len(); i++) {
        pos[i] = transform * vec4(pos[i], 1.);
    }

    MeshParams mp = {
        .indices = &indices,
        .pos = &pos,
        .material_id = mat_id,
        .light_id = light_id,
    };

    rc->add_mesh(mp);
}

void SceneLoader::load_obj(pugi::xml_node shape_node, u32 mat_id, const mat4 &transform,
                           cuda::std::optional<u32> light_id, RenderContext *rc) {
    std::string filename = shape_node.child("string").attribute("value").as_string();
    auto file_path = scene_base_path + "/" + filename;

    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./";
    reader_config.triangulate = true;

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(file_path, reader_config)) {
        if (!reader.Error().empty()) {
            spdlog::error("Error reading obj file: '{}'", reader.Error());
        }
        throw;
    }

    if (!reader.Warning().empty()) {
        spdlog::warn("Warning when reading obj file");
    }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();

    SharedVector<vec3> pos{};
    SharedVector<vec3> normals{};
    SharedVector<vec2> uvs{};
    SharedVector<u32> indices{};

    assert(shapes.size() == 1);
    auto shape = &shapes[0];

    // Wavefront OBJ format is so cursed....
    // Load indices
    for (size_t f = 0; f < shape->mesh.num_face_vertices.size(); f++) {
        auto fv = size_t(shape->mesh.num_face_vertices[f]);
        // Triangulation should take care of this...
        assert(fv == 3);

        int i0 = shape->mesh.indices[3 * f].vertex_index;
        int i1 = shape->mesh.indices[3 * f + 1].vertex_index;
        int i2 = shape->mesh.indices[3 * f + 2].vertex_index;

        for (int i = 0; i < 3; i++) {
            auto ind = shape->mesh.indices[3 * f + i];
            assert(ind.vertex_index == ind.normal_index &&
                   ind.vertex_index == ind.texcoord_index &&
                   ind.vertex_index == ind.texcoord_index);
        }

        indices.push(std::move(i0));
        indices.push(std::move(i1));
        indices.push(std::move(i2));
    }

    size_t pos_size = attrib.vertices.size() / 3;
    size_t normals_size = attrib.normals.size() / 3;
    size_t uvs_size = attrib.texcoords.size() / 2;

    assert(pos_size == normals_size && pos_size == uvs_size);

    // Copy vertices
    for (int v = 0; v < pos_size; v++) {
        tinyobj::real_t x = attrib.vertices[3 * v];
        tinyobj::real_t y = attrib.vertices[3 * v + 1];
        tinyobj::real_t z = attrib.vertices[3 * v + 2];

        vec3 vert_pos = vec3(x, y, z);
        pos.push(std::move(vert_pos));

        tinyobj::real_t nx = attrib.normals[3 * v];
        tinyobj::real_t ny = attrib.normals[3 * v + 1];
        tinyobj::real_t nz = attrib.normals[3 * v + 2];

        vec3 vert_normal = vec3(nx, ny, nz);
        normals.push(std::move(vert_normal));

        tinyobj::real_t tx = attrib.texcoords[2 * v];
        tinyobj::real_t ty = attrib.texcoords[2 * v + 1];

        vec2 uv = vec2(tx, ty);
        uvs.push(std::move(uv));
    }

    for (int i = 0; i < pos.len(); i++) {
        pos[i] = transform * vec4(pos[i], 1.);
    }

    auto inv_trans = glm::inverse(glm::transpose(transform));
    for (int i = 0; i < normals.len(); i++) {
        normals[i] = inv_trans * vec4(normals[i], 1.);
    }

    MeshParams mp = {
        .indices = &indices,
        .pos = &pos,
        .normals = &normals,
        .uvs = &uvs,
        .material_id = mat_id,
        .light_id = light_id,
    };

    rc->add_mesh(mp);
}

std::optional<SceneAttribs> SceneLoader::load_scene_attribs() {
    SceneAttribs attribs;
    auto scene = doc.child("scene");

    for (pugi::xml_node def : scene.children("default")) {
        std::string name = def.attribute("name").as_string();
        if (name == "resx") {
            attribs.resx = def.attribute("value").as_int();
        } else if (name == "resy") {
            attribs.resy = def.attribute("value").as_int();
        }
    }

    auto sensor = scene.child("sensor");
    for (pugi::xml_node f : sensor.children("float")) {
        std::string name = f.attribute("name").as_string();
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
