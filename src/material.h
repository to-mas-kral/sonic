#ifndef PT_MATERIAL_H
#define PT_MATERIAL_H

#include "color/spectrum.h"
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
    make_diffuse(const RgbSpectrum &p_reflectance) {
        return Material{.type = MaterialType::Diffuse,
                        .diffuse = {
                            .reflectance = p_reflectance,
                            .reflectance_tex_id = {},
                        }};
    }

    static Material
    make_diffuse(u32 reflectance_tex_id) {
        return Material{.type = MaterialType::Diffuse,
                        .diffuse = {
                            .reflectance = RgbSpectrum{},
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

    __device__ spectral
    eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas, Texture *textures,
         const vec2 &uv) const {
        switch (type) {
        case MaterialType::Diffuse: {
            spectral refl{};

            if (diffuse.reflectance_tex_id.has_value()) {
                auto tex_id = diffuse.reflectance_tex_id.value();
                auto texture = &textures[tex_id];

                auto sigmoid_coeff = texture->fetch(uv);
                refl = RgbSpectrum::from_coeff(sigmoid_coeff).eval(lambdas);
            } else {
                refl = diffuse.reflectance.eval(lambdas);
            }

            return refl / M_PIf;
        }
        case MaterialType::Conductor:
            return spectral::ONE() / abs(sgeom.cos_theta);
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
            RgbSpectrum reflectance;
            COption<u32> reflectance_tex_id = {};
        } diffuse;
    };
};

#endif // PT_MATERIAL_H
