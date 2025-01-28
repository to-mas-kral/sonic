#ifndef PT_MATERIAL_H
#define PT_MATERIAL_H

#include "../color/spectral_quantity.h"
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

class Material {
public:
    explicit Material(const DiffuseMaterial &diffuse_material);

    explicit Material(const DiffuseTransmissionMaterial &diffuse_transmission);

    explicit Material(const CoatedDifuseMaterial &coated_difuse_material);

    explicit Material(const RoughCoatedDiffuseMaterial &rough_coated_diffuse_material);

    explicit Material(const DielectricMaterial &dielectric_material);

    explicit Material(const ConductorMaterial &conductor_material);

    explicit Material(const RoughConductorMaterial &rough_conductor_material);

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

    bool
    is_translucent() const;

    bool is_twosided = false;

private:
    MaterialType type;
    union {
        DiffuseMaterial diffuse;
        DiffuseTransmissionMaterial diffuse_transmission;
        CoatedDifuseMaterial coated_diffuse;
        RoughCoatedDiffuseMaterial rough_coated_diffuse;
        DielectricMaterial dielectric;
        ConductorMaterial conductor;
        RoughConductorMaterial rough_conductor;
    };
};

#endif // PT_MATERIAL_H
