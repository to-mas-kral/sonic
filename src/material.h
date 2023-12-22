#ifndef PT_MATERIAL_H
#define PT_MATERIAL_H

#include "integrator/utils.h"
#include "math/sampling.h"
#include "math/vecmath.h"
#include "utils/basic_types.h"

enum class MaterialType : u8 {
    Diffuse = 0,
    Conductor = 1,
};

struct Material {
    static Material
    make_diffuse(const vec3 &p_reflectance) {
        return Material{
            .type = MaterialType::Diffuse,
            .diffuse = {
                .reflectance = {p_reflectance.x, p_reflectance.y, p_reflectance.z},
                .reflectance_tex_id = {},
            }};
    }

    static Material
    make_diffuse(u32 reflectance_tex_id) {
        return Material{.type = MaterialType::Diffuse,
                        .diffuse = {
                            .reflectance = {0.f, 0.f, 0.f},
                            .reflectance_tex_id = reflectance_tex_id,
                        }};
    }

    static Material
    make_conductor() {
        return Material{
            .type = MaterialType::Conductor,
        };
    }

    __device__ norm_vec3
    sample(const norm_vec3 &normal, const norm_vec3 &wo, const vec2 &sample) const {
        switch (type) {
        case MaterialType::Diffuse: {
            norm_vec3 sample_dir = sample_cosine_hemisphere(sample);
            return orient_dir(sample_dir, normal);
        }
        case MaterialType::Conductor: {
            return vec3::reflect(wo, normal).normalized();
        }
        }
    }

    // Probability density function of sampling the BRDF
    __device__ f32
    pdf(const ShadingGeometry &sgeom, bool was_generated = false) const {
        switch (type) {
        case MaterialType::Diffuse: {
            return sgeom.cos_theta / M_PIf;
        }
        case MaterialType::Conductor: {
            if (was_generated) {
                return 1.f;
            } else {
                return 0.f;
            }
        }
        }
    }

    __device__ vec3
    eval(const ShadingGeometry &sgeom, Texture *textures, const vec2 &uv) const {
        switch (type) {
        case MaterialType::Diffuse: {
            vec3 refl = vec3(diffuse.reflectance[0], diffuse.reflectance[1],
                             diffuse.reflectance[2]);
            if (diffuse.reflectance_tex_id.has_value()) {
                auto tex_id = diffuse.reflectance_tex_id.value();
                auto texture = &textures[tex_id];
                refl = vec3(texture->fetch(uv));
            }

            return refl / M_PIf;
        }
        case MaterialType::Conductor:
            return vec3(1.f) / abs(sgeom.cos_theta);
        }
    }

    __device__ bool
    is_specular() const {
        switch (type) {
        case MaterialType::Diffuse:
            return false;
        case MaterialType::Conductor:
            return true;
        }
    }

    MaterialType type = MaterialType::Diffuse;
    bool is_twosided = false;
    union {
        struct {
            f32 reflectance[3] = {0.5f, 0.5f, 0.5f};
            COption<u32> reflectance_tex_id = {};
        } diffuse;
    };
};

#endif // PT_MATERIAL_H
