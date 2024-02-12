#ifndef PT_MATERIAL_H
#define PT_MATERIAL_H

#include "../color/sampled_spectrum.h"
#include "../color/spectrum.h"
#include "../integrator/utils.h"
#include "../texture.h"
#include "../utils/basic_types.h"
#include "bsdf_sample.h"
#include "common.h"
#include "conductor.h"
#include "dielectric.h"
#include "diffuse.h"
#include "plastic.h"
#include "rough_conductor.h"
#include "rough_plastic.h"

enum class MaterialType : u8 {
    Diffuse,
    Plastic,
    RoughPlastic,
    Conductor,
    RoughConductor,
    Dielectric,
};

struct Material {
    static Material
    make_diffuse(u32 reflectance_tex_id) {
        return Material{.type = MaterialType::Diffuse,
                        .diffuse = {
                            .reflectance_tex_id = reflectance_tex_id,
                        }};
    }

    static Material
    make_dielectric(Spectrum ext_ior, Spectrum int_ior, Spectrum transmittance) {
        return Material{.type = MaterialType::Dielectric,
                        .dielectric = {
                            .m_int_ior = int_ior,
                            .m_ext_ior = ext_ior,
                            .m_transmittance = transmittance,
                        }};
    }

    static Material
    make_conductor(Spectrum eta, Spectrum k) {
        return Material{.type = MaterialType::Conductor,
                        .conductor = {
                            .m_perfect = false,
                            .m_eta = eta,
                            .m_k = k,
                        }};
    }

    static Material
    make_conductor_perfect() {
        return Material{.type = MaterialType::Conductor,
                        .conductor = {
                            .m_perfect = true,
                            .m_eta = Spectrum(ConstantSpectrum::make(0.f)),
                            .m_k = Spectrum(ConstantSpectrum::make(0.f)),
                        }};
    }

    static Material
    make_rough_conductor(f32 alpha, Spectrum eta, Spectrum k) {
        if (TrowbridgeReitzGGX::is_alpha_effectively_zero(alpha)) {
            return Material{.type = MaterialType::Conductor,
                            .conductor = {
                                .m_perfect = false,
                                .m_eta = eta,
                                .m_k = k,
                            }};
        } else {
            return Material{.type = MaterialType::RoughConductor,
                            .rough_conductor = {
                                .m_eta = eta,
                                .m_k = k,
                                .m_alpha = alpha,
                            }};
        }
    }

    static Material
    make_plastic(Spectrum ext_ior, Spectrum int_ior, u32 diffuse_reflectance_id) {
        return Material{.type = MaterialType::Plastic,
                        .plastic = {
                            .ext_ior = ext_ior,
                            .int_ior = int_ior,
                            .diffuse_reflectance_id = diffuse_reflectance_id,
                        }};
    }

    static Material
    make_rough_plastic(f32 alpha, Spectrum ext_ior, Spectrum int_ior,
                       u32 diffuse_reflectance_id) {
        if (TrowbridgeReitzGGX::is_alpha_effectively_zero(alpha)) {
            return Material{.type = MaterialType::Plastic,
                            .plastic = {
                                .ext_ior = ext_ior,
                                .int_ior = int_ior,
                                .diffuse_reflectance_id = diffuse_reflectance_id,
                            }};
        } else {
            return Material{.type = MaterialType::RoughPlastic,
                            .rough_plastic = {
                                .alpha = alpha,
                                .ext_ior = ext_ior,
                                .int_ior = int_ior,
                                .diffuse_reflectance_id = diffuse_reflectance_id,
                            }};
        }
    }

    __device__ COption<BSDFSample>
    sample(const norm_vec3 &normal, const norm_vec3 &wo, const vec3 &sample,
           const SampledLambdas &lambdas, const Texture *textures, const vec2 &uv,
           bool is_frontfacing) const {
        switch (type) {
        case MaterialType::Diffuse:
            return diffuse.sample(normal, wo, vec2(sample.x, sample.y), lambdas, textures,
                                  uv);
        case MaterialType::Plastic:
            return plastic.sample(normal, wo, sample, lambdas, textures, uv);
        case MaterialType::RoughPlastic:
            return rough_plastic.sample(normal, wo, sample, lambdas, textures, uv);
        case MaterialType::Conductor:
            return conductor.sample(normal, wo, lambdas, textures, uv);
        case MaterialType::RoughConductor:
            return rough_conductor.sample(normal, wo, vec2(sample.x, sample.y), lambdas,
                                          textures, uv);
        case MaterialType::Dielectric:
            return dielectric.sample(normal, wo, vec2(sample.x, sample.y), lambdas,
                                     textures, uv, is_frontfacing);
        }
    }

    // Probability density function of sampling the BRDF
    __host__ __device__ f32
    pdf(const ShadingGeometry &sgeom, const SampledLambdas &λ) const {
        switch (type) {
        case MaterialType::Diffuse:
            return DiffuseMaterial::pdf(sgeom);
        case MaterialType::Plastic:
            return plastic.pdf(sgeom, λ);
        case MaterialType::RoughPlastic:
            return rough_plastic.pdf(sgeom, λ);
        case MaterialType::Conductor:
            return ConductorMaterial::pdf();
        case MaterialType::RoughConductor:
            return rough_conductor.pdf(sgeom);
        case MaterialType::Dielectric:
            return DielectricMaterial::pdf();
        }
    }

    __device__ spectral
    eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
         const Texture *textures, const vec2 &uv) const {
        switch (type) {
        case MaterialType::Diffuse:
            return diffuse.eval(sgeom, lambdas, textures, uv);
        case MaterialType::Plastic:
            return plastic.eval(sgeom, lambdas, textures, uv);
        case MaterialType::RoughPlastic:
            return rough_plastic.eval(sgeom, lambdas, textures, uv);
        case MaterialType::RoughConductor:
            return rough_conductor.eval(sgeom, lambdas);
        case MaterialType::Conductor:
            return conductor.eval(sgeom, lambdas, textures, uv);
        case MaterialType::Dielectric:
            return DielectricMaterial::eval();
        }
    }

    __device__ bool
    is_dirac_delta() const {
        switch (type) {
        case MaterialType::Diffuse:
            return false;
        case MaterialType::Plastic:
            return PlasticMaterial::is_dirac_delta();
        case MaterialType::RoughPlastic:
            return false;
        case MaterialType::Conductor:
            return true;
        case MaterialType::RoughConductor:
            return false;
        case MaterialType::Dielectric:
            return true;
        }
    }

    // TODO: discriminated ptr would be nice here...
    MaterialType type = MaterialType::Diffuse;
    bool is_twosided = false;
    union {
        DiffuseMaterial diffuse;
        PlasticMaterial plastic;
        RoughPlasticMaterial rough_plastic;
        DielectricMaterial dielectric;
        ConductorMaterial conductor;
        RoughConductorMaterial rough_conductor;
    };
};

#endif // PT_MATERIAL_H
