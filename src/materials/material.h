#ifndef PT_MATERIAL_H
#define PT_MATERIAL_H

#include "../color/sampled_spectrum.h"
#include "../color/spectrum.h"
#include "../scene/texture.h"
#include "../utils/basic_types.h"
#include "bsdf_sample.h"
#include "coateddiffuse.h"
#include "conductor.h"
#include "dielectric.h"
#include "diffuse.h"
#include "diffuse_transmission.h"
#include "rough_coateddiffuse.h"
#include "rough_conductor.h"

enum class MaterialType : u8 {
    Diffuse,
    DiffuseTransmission,
    CoatedDiffuse,
    RoughCoatedDiffuse,
    Conductor,
    RoughConductor,
    Dielectric,
};

struct Material {
    static Material
    make_diffuse(SpectrumTexture *reflectance);

    static Material
    make_diffuse_transmission(SpectrumTexture *reflectance,
                              SpectrumTexture *transmittance, f32 scale);

    static Material
    make_dielectric(const Spectrum &ext_ior, SpectrumTexture *int_ior,
                    const Spectrum &transmittance);

    static Material
    make_conductor(SpectrumTexture *eta, SpectrumTexture *k);

    static Material
    make_rough_conductor(FloatTexture *alpha, SpectrumTexture *eta, SpectrumTexture *k);

    static Material
    make_plastic(const Spectrum &ext_ior, const Spectrum &int_ior,
                 SpectrumTexture *diffuse_reflectance);

    static Material
    make_rough_plastic(FloatTexture *alpha, const Spectrum &ext_ior,
                       const Spectrum &int_ior, SpectrumTexture *diffuse_reflectance);

    std::optional<BSDFSample>
    sample(const ShadingFrameIncomplete &sframe, norm_vec3 wo, const vec3 &xi,
           SampledLambdas &lambdas, const vec2 &uv, bool is_frontfacing) const;

    // Probability density function of sampling the BRDF
    f32
    pdf(const ShadingFrame &sframe, const SampledLambdas &lambdas, const vec2 &uv) const;

    spectral
    eval(const ShadingFrame &sframe, const SampledLambdas &lambdas, const vec2 &uv) const;

    bool
    is_delta() const;

    MaterialType type = MaterialType::Diffuse;
    bool is_twosided = false;

    union {
        DiffuseMaterial diffuse;
        DiffuseTransmissionMaterial diffusetransmission;
        CoatedDifuseMaterial coateddiffuse;
        RoughCoatedDiffuseMaterial rough_coateddiffuse;
        DielectricMaterial dielectric;
        ConductorMaterial conductor;
        RoughConductorMaterial rough_conductor;
    };
};

#endif // PT_MATERIAL_H
