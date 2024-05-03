#include "mitsuba_loader.h"

#include <ranges>
#include <utility>

#include "../color/spectral_data.h"
#include <fmt/core.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using str = std::string_view;

pugi::xml_node
child_node(const pugi::xml_node parent, const std::string &name) {
    auto node = parent.find_child([&](pugi::xml_node node) {
        return node.find_attribute([&](pugi::xml_attribute attr) {
            return str(attr.name()) == "name" && str(attr.as_string()) == name;
        });
    });

    return node;
}

pugi::xml_attribute
child_node_attr(const pugi::xml_node parent, const std::string &node_name,
                const std::string &attr) {
    auto node = parent.find_child([&](pugi::xml_node node) {
        return node.find_attribute([&](pugi::xml_attribute attr) {
            return str(attr.name()) == "name" && str(attr.as_string()) == node_name;
        });
    });

    return node.attribute(attr.data());
}

void
MitsubaLoader::load_scene(Scene &sc) {
    load_scene_attribs(sc.attribs);

    auto scene = doc.child("scene");

    load_materials(scene, sc);
    load_shapes(sc, scene);

    auto envmap_node = scene.child("emitter");
    if (envmap_node) {
        std::string filename = envmap_node.child("string").attribute("value").as_string();
        auto file_path = scene_base_path + "/" + filename;

        auto transform_node = envmap_node.child("transform");
        auto to_world_transform = parse_transform(transform_node);

        auto envmap = Envmap(file_path, to_world_transform);
        sc.set_envmap(std::move(envmap));
    }
}

void
MitsubaLoader::load_shapes(Scene &sc, const pugi::xml_node &scene) {
    for (pugi::xml_node shape : scene.children("shape")) {
        str type = shape.attribute("type").as_string();

        MaterialId mat_id{0};

        auto ref_node = shape.child("ref");
        auto bsdf_node = shape.child("bsdf");
        if (ref_node) {
            std::string bsdf_id = ref_node.attribute("id").as_string();
            mat_id = materials_cache.at(bsdf_id);
        } else if (bsdf_node) {
            auto [mat, _] = load_material(sc, bsdf_node);
            mat_id = sc.add_material(mat);
        } else {
            throw std::runtime_error("Shape has no material");
        }

        auto transform_node = shape.child("transform");
        mat4 transform = mat4::identity();
        if (transform_node) {
            transform = parse_transform(transform_node);
        }

        auto emitter_node = shape.child("emitter");
        Option<Emitter> emitter = {};
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
MitsubaLoader::load_materials(pugi::xml_node scene, Scene &sc) {
    // HACK: default material
    auto texture = Texture::make_constant_texture(RgbSpectrum::make(tuple3(0.5f)));
    TextureId tex_id = sc.add_texture(texture);

    auto bsdfs = scene.children("bsdf");

    for (auto bsdf : bsdfs) {
        auto [mat, id] = load_material(sc, bsdf);

        MaterialId mat_id = sc.add_material(mat);
        materials_cache.insert({id, mat_id});
    }
}

Tuple<Material, std::string>
MitsubaLoader::load_material(Scene &scene, pugi::xml_node &bsdf) {
    str type = bsdf.attribute("type").as_string();
    std::string id = bsdf.attribute("id").as_string();
    bool is_twosided = false;

    while (type == "twosided" || type == "bumpmap") {
        if (type == "twosided") {
            is_twosided = true;
        }

        bsdf = bsdf.child("bsdf");
        type = bsdf.attribute("type").as_string();

        // For bumpmaps, the id is in the nested bsdf... ??
        if (id.empty()) {
            id = bsdf.attribute("id").as_string();
        }
    }

    // Default material - 50% reflectance
    Material mat = Material::make_diffuse(TextureId{0});

    if (type == "diffuse") {
        mat = load_diffuse_material(scene, bsdf);
        mat.is_twosided = is_twosided;
    } else if (type == "plastic") {
        mat = load_plastic_material(scene, bsdf);
        mat.is_twosided = is_twosided;
    } else if (type == "roughplastic") {
        mat = load_roughp_lastic_material(scene, bsdf);
        mat.is_twosided = is_twosided;
    } else if (type == "conductor") {
        mat = load_conductor_material(scene, bsdf);
        mat.is_twosided = is_twosided;
    } else if (type == "dielectric") {
        mat = load_dielectric_material(scene, bsdf);
        mat.is_twosided = true;
    } else if (type == "roughconductor") {
        mat = load_roughconductor_material(scene, bsdf);
        mat.is_twosided = is_twosided;
    } else {
        spdlog::warn("Unknown BSDF type: {}, defaulting to diffuse", type);
        mat.is_twosided = is_twosided;
    }

    return {mat, id};
}

Material
MitsubaLoader::load_plastic_material(Scene &sc, const pugi::xml_node &bsdf) const {
    Spectrum int_ior(POLYPROPYLENE_ETA);
    Spectrum ext_ior(AIR_ETA);

    auto int_ior_node = child_node(bsdf, "int_ior");
    if (int_ior_node) {
        int_ior =
            Spectrum(ConstantSpectrum::make(int_ior_node.attribute("value").as_float()));
    }

    auto ext_ior_node = child_node(bsdf, "ext_ior");
    if (ext_ior_node) {
        ext_ior =
            Spectrum(ConstantSpectrum::make(ext_ior_node.attribute("value").as_float()));
    }

    auto reflectance_node = child_node(bsdf, "reflectance");
    auto diffuse_reflectance_id = load_texture(sc, reflectance_node);
    return Material::make_plastic(ext_ior, int_ior, diffuse_reflectance_id,
                                  sc.material_allocator);
}

Material
MitsubaLoader::load_roughp_lastic_material(Scene &sc, const pugi::xml_node &bsdf) const {
    Spectrum int_ior(POLYPROPYLENE_ETA);
    Spectrum ext_ior(AIR_ETA);

    // TODO: nonlinear default is false

    auto int_ior_node = child_node(bsdf, "int_ior");
    if (int_ior_node) {
        int_ior =
            Spectrum(ConstantSpectrum::make(int_ior_node.attribute("value").as_float()));
    }

    auto ext_ior_node = child_node(bsdf, "ext_ior");
    if (ext_ior_node) {
        ext_ior =
            Spectrum(ConstantSpectrum::make(ext_ior_node.attribute("value").as_float()));
    }

    Spectrum diffuse_reflectance(ConstantSpectrum::make(0.5f));

    auto reflectance_node = child_node(bsdf, "reflectance");
    auto diffuse_reflectance_id = load_texture(sc, reflectance_node);

    f32 alpha = child_node_attr(bsdf, "alpha", "value").as_float();
    const auto alpha_tex = sc.add_texture(Texture::make_constant_texture(alpha));

    return Material::make_rough_plastic(alpha_tex, ext_ior, int_ior,
                                        diffuse_reflectance_id, sc.material_allocator);
}

Material
MitsubaLoader::load_conductor_material(Scene &sc, const pugi::xml_node &bsdf) const {
    auto mat_node = child_node(bsdf, "material");
    if (mat_node) {
        if (mat_node.attribute("value").as_string() == str("none")) {
            return Material::make_conductor_perfect(sc.material_allocator);
        } else {
            throw std::runtime_error(
                "Named material for rough conductors aren't implemented yet");
        }
    }

    tuple3 eta = parse_tuple3(child_node_attr(bsdf, "eta", "value").as_string());
    tuple3 k = parse_tuple3(child_node_attr(bsdf, "k", "value").as_string());

    auto eta_spectrum = RgbSpectrumUnbounded::make(eta);
    auto k_spectrum = RgbSpectrumUnbounded::make(k);

    auto eta_tex = sc.add_texture(Texture::make_constant_texture(eta_spectrum));
    auto k_tex = sc.add_texture(Texture::make_constant_texture(k_spectrum));

    return Material::make_conductor(eta_tex, k_tex, sc.material_allocator);
}

Material
MitsubaLoader::load_roughconductor_material(Scene &sc, const pugi::xml_node &bsdf) const {
    f32 alpha = child_node_attr(bsdf, "alpha", "value").as_float();
    tuple3 eta = parse_tuple3(child_node_attr(bsdf, "eta", "value").as_string());
    tuple3 k = parse_tuple3(child_node_attr(bsdf, "k", "value").as_string());

    auto eta_spectrum = RgbSpectrumUnbounded::make(eta);
    auto k_spectrum = RgbSpectrumUnbounded::make(k);

    auto eta_tex = sc.add_texture(Texture::make_constant_texture(eta_spectrum));
    auto k_tex = sc.add_texture(Texture::make_constant_texture(k_spectrum));

    auto alpha_tex = sc.add_texture(Texture::make_constant_texture(alpha));

    return Material::make_rough_conductor(alpha_tex, eta_tex, k_tex,
                                          sc.material_allocator);
}

Material
MitsubaLoader::load_dielectric_material(Scene &sc, const pugi::xml_node &bsdf) const {
    Spectrum int_ior(GLASS_BK7_ETA, &sc.spectrum_allocator);
    Spectrum ext_ior(AIR_ETA);

    auto int_ior_node = child_node(bsdf, "int_ior");
    if (int_ior_node) {
        int_ior =
            Spectrum(ConstantSpectrum::make(int_ior_node.attribute("value").as_float()));
    }

    auto ext_ior_node = child_node(bsdf, "ext_ior");
    if (ext_ior_node) {
        ext_ior =
            Spectrum(ConstantSpectrum::make(ext_ior_node.attribute("value").as_float()));
    }

    auto transmittance_node = child_node(bsdf, "specular_transmittance");

    Spectrum specular_transmittance(ConstantSpectrum::make(1.f));
    if (transmittance_node) {
        tuple3 rgb = parse_tuple3(transmittance_node.attribute("value").as_string());
        specular_transmittance = Spectrum(RgbSpectrum::make(rgb));
    }

    auto int_ior_tex =
        sc.add_texture(Texture::make_constant_texture(int_ior.eval_single(400.f)));

    return Material::make_dielectric(ext_ior, int_ior_tex, specular_transmittance,
                                     sc.material_allocator);
}

TextureId
MitsubaLoader::load_texture(Scene &sc, const pugi::xml_node &texture_node) const {
    if (str(texture_node.name()) == "texture") {
        auto filename_node = child_node(texture_node, "filename");
        auto file_name = filename_node.attribute("value").as_string();
        auto file_path = this->scene_base_path + "/" + file_name;

        auto texture = Texture::make_image_texture(file_path, true);

        auto tex_id = sc.add_texture(texture);
        return tex_id;
    } else {
        tuple3 rgb = parse_tuple3(texture_node.attribute("value").as_string());
        auto texture = Texture::make_constant_texture(RgbSpectrum::make(rgb));
        auto tex_id = sc.add_texture(texture);
        return tex_id;
    }
}

Material
MitsubaLoader::load_diffuse_material(Scene &sc, const pugi::xml_node &bsdf) {
    auto reflectance_node = child_node(bsdf, "reflectance");
    auto tex_id = load_texture(sc, reflectance_node);
    return Material::make_diffuse(tex_id);
}

mat4
MitsubaLoader::parse_transform(pugi::xml_node transform_node) {
    mat4 composed_transform = mat4::identity();

    for (auto subtransform_node : transform_node.children()) {
        str name = subtransform_node.name();

        mat4 transform;
        if (name == "matrix") {
            transform = parse_transform_matrix(subtransform_node);
        } else if (name == "rotate") {
            transform = parse_transform_rotate(subtransform_node);
        } else {
            throw std::runtime_error(fmt::format("Unknown transform type: {}", name));
        }

        composed_transform = composed_transform.compose(transform);
    }

    return composed_transform;
}

mat4
MitsubaLoader::parse_transform_rotate(const pugi::xml_node &transform_node) {
    mat4 transform;
    auto angle_attr = transform_node.attribute("angle");
    if (angle_attr) {
        f32 angle = to_rad(angle_attr.as_float());

        if (transform_node.attribute("y")) {
            transform = mat4::from_euler_y(angle);
        } else if (transform_node.attribute("x")) {
            transform = mat4::from_euler_x(angle);
        } else if (transform_node.attribute("z")) {
            transform = mat4::from_euler_z(angle);
        }
    } else {
        throw std::runtime_error("Rotate along arbitrary axis not implemented");
    }

    return transform;
}

mat4
MitsubaLoader::parse_transform_matrix(const pugi::xml_node &matrix_node) {
    str matrix_str = matrix_node.attribute("value").as_string();

    auto floats = std::views::transform(std::views::split(matrix_str, ' '), [](auto v) {
        auto c = v | std::views::common;
        return std::stof(std::string(c.begin(), c.end()));
    });

    Array<f32, 16> mat{};
    int i = 0;
    for (f32 f : floats) {
        if (i > 15) {
            throw std::runtime_error("Wrong matrix element count");
        }

        mat[i] = f;

        i++;
    }

    // matrices are stored in column-majorm, but Mitsuba's format is row-major...
    return mat4::from_elements(mat).transpose();
}

tuple3
MitsubaLoader::parse_tuple3(const std::string &str) {
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
MitsubaLoader::load_emitter(pugi::xml_node emitter_node, Scene &sc) {
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

    return Emitter(Spectrum(emittance, &sc.spectrum_allocator), false);
}

void
MitsubaLoader::load_rectangle(pugi::xml_node shape, MaterialId mat_id,
                              const mat4 &transform, Option<Emitter> emitter, Scene &sc) {
    // clang-format off
    std::vector<point3> pos = {
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
    std::vector<u32> indices{
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

    sc.add_mesh(mp);
}

void
MitsubaLoader::load_cube(pugi::xml_node shape, MaterialId mat_id, const mat4 &transform,
                         Option<Emitter> emitter, Scene &sc) {
    // clang-format off
    std::vector<point3> pos = {
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
    std::vector<u32> indices{
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

    sc.add_mesh(mp);
}

void
MitsubaLoader::load_sphere(pugi::xml_node node, MaterialId mat_id, mat4 transform,
                           Option<Emitter> emitter, Scene &sc) {
    auto radius_node = child_node(node, "radius");
    f32 radius = radius_node.attribute("value").as_float();
    auto center_node = child_node(node, "center");

    f32 x = center_node.attribute("x").as_float();
    f32 y = center_node.attribute("y").as_float();
    f32 z = center_node.attribute("z").as_float();

    point3 center = point3(x, y, z);
    auto sphere = SphereParams{center, radius, mat_id, emitter};
    sc.add_sphere(sphere);
}

void
MitsubaLoader::load_obj(pugi::xml_node shape_node, MaterialId mat_id,
                        const mat4 &transform, Option<Emitter> emitter, Scene &sc) {
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
    auto face_normals_node = child_node(shape_node, "face_normals");
    if (face_normals_node) {
        face_normals = face_normals_node.attribute("value").as_bool();
    }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();

    std::vector<point3> pos{};
    std::vector<vec3> normals{};
    std::vector<vec2> uvs{};
    std::vector<u32> indices{};

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

        indices.push_back(i0);
        indices.push_back(i1);
        indices.push_back(i2);
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
        pos.push_back(vert_pos);

        if (!face_normals) {
            tinyobj::real_t nx = attrib.normals[3 * v];
            tinyobj::real_t ny = attrib.normals[3 * v + 1];
            tinyobj::real_t nz = attrib.normals[3 * v + 2];

            vec3 vert_normal = vec3(nx, ny, nz);
            normals.push_back(vert_normal);
        }

        tinyobj::real_t tx = attrib.texcoords[2 * v];
        tinyobj::real_t ty = attrib.texcoords[2 * v + 1];

        vec2 uv = vec2(tx, ty);
        uvs.push_back(uv);
    }

    for (auto &po : pos) {
        po = transform.transform_point(po);
    }

    auto inv_trans = transform.transpose().inverse();
    for (auto &normal : normals) {
        normal = inv_trans.transform_vec(normal);
    }

    MeshParams mp = {
        .indices = &indices,
        .pos = &pos,
        .normals = (face_normals) ? nullptr : &normals,
        .uvs = &uvs,
        .material_id = mat_id,
        .emitter = emitter,
    };

    sc.add_mesh(mp);
}

SceneAttribs
MitsubaLoader::load_scene_attribs(SceneAttribs &attribs) {
    auto scene = doc.child("scene");

    for (pugi::xml_node def : scene.children("default")) {
        str name = def.attribute("name").as_string();
        if (name == "resx") {
            attribs.film.resx = def.attribute("value").as_uint();
        } else if (name == "resy") {
            attribs.film.resy = def.attribute("value").as_uint();
        } else if (name == "max_depth") {
            attribs.max_depth = def.attribute("value").as_uint();
        }
    }

    auto sensor = scene.child("sensor");
    for (pugi::xml_node f : sensor.children("float")) {
        str name = f.attribute("name").as_string();
        if (name == "fov") {
            attribs.camera.fov = f.attribute("value").as_float();
        }
    }

    auto transform = sensor.child("transform");
    if (transform) {
        attribs.camera.camera_to_world = parse_transform(transform);
    }

    return attribs;
}
