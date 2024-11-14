#include "material.h"

#include "../integrator/shading_frame.h"
#include "diffuse_transmission.h"

Material::
Material(const DiffuseMaterial &diffuse_material)
    : type{MaterialType::Diffuse}, diffuse{diffuse_material} {}

Material::
Material(const DiffuseTransmissionMaterial &diffuse_transmission)
    : type{MaterialType::DiffuseTransmission},
      diffuse_transmission{diffuse_transmission} {}

Material::
Material(const CoatedDifuseMaterial &coated_difuse_material)
    : type{MaterialType::CoatedDiffuse}, coated_diffuse{coated_difuse_material} {}

Material::
Material(const RoughCoatedDiffuseMaterial &rough_coated_diffuse_material)
    : type{MaterialType::RoughCoatedDiffuse},
      rough_coated_diffuse{rough_coated_diffuse_material} {}

Material::
Material(const DielectricMaterial &dielectric_material)
    : type{MaterialType::Dielectric}, dielectric{dielectric_material} {}

Material::
Material(const ConductorMaterial &conductor_material)
    : type{MaterialType::Conductor}, conductor{conductor_material} {}

Material::
Material(const RoughConductorMaterial &rough_conductor_material)
    : type{MaterialType::RoughConductor}, rough_conductor{rough_conductor_material} {}

std::optional<BSDFSample>
Material::sample(const ShadingFrameIncomplete &sframe, norm_vec3 wo, const vec3 &xi,
                 SampledLambdas &lambdas, const vec2 &uv,
                 const bool is_frontfacing) const {
    wo = sframe.to_local(wo).normalized();

    if (sframe.cos_theta(wo) == 0.F) {
        return {};
    }

    std::optional<BSDFSample> bsdf_sample{};

    switch (type) {
    case MaterialType::Diffuse:
        bsdf_sample = diffuse.sample(sframe, wo, vec2(xi.x, xi.y), lambdas, uv);
        break;
    case MaterialType::DiffuseTransmission:
        bsdf_sample =
            diffuse_transmission.sample(sframe, wo, vec2(xi.x, xi.y), lambdas, uv);
        break;
    case MaterialType::CoatedDiffuse:
        bsdf_sample = coated_diffuse.sample(sframe, wo, xi, lambdas, uv);
        break;
    case MaterialType::RoughCoatedDiffuse:
        bsdf_sample = rough_coated_diffuse.sample(sframe, wo, xi, lambdas, uv);
        break;
    case MaterialType::Conductor:
        bsdf_sample = conductor.sample(sframe, wo, lambdas, uv);
        break;
    case MaterialType::RoughConductor:
        bsdf_sample = rough_conductor.sample(sframe, wo, vec2(xi.x, xi.y), lambdas, uv);
        break;
    case MaterialType::Dielectric:
        bsdf_sample =
            dielectric.sample(sframe, wo, vec2(xi.x, xi.y), lambdas, uv, is_frontfacing);
        break;
    default:
        panic();
    }

    if (bsdf_sample.has_value()) {
        assert(!bsdf_sample->bsdf.is_invalid());
    }

    return bsdf_sample;
}

f32
Material::pdf(const ShadingFrame &sframe, const SampledLambdas &lambdas,
              const vec2 &uv) const {
    f32 pdf{};

    switch (type) {
    case MaterialType::Diffuse:
        pdf = DiffuseMaterial::pdf(sframe);
        break;
    case MaterialType::DiffuseTransmission:
        pdf = DiffuseTransmissionMaterial::pdf(sframe);
        break;
    case MaterialType::CoatedDiffuse:
        pdf = coated_diffuse.pdf(sframe, lambdas);
        break;
    case MaterialType::RoughCoatedDiffuse:
        pdf = rough_coated_diffuse.pdf(sframe, lambdas, uv);
        break;
    case MaterialType::Conductor:
        pdf = ConductorMaterial::pdf();
        break;
    case MaterialType::RoughConductor:
        pdf = rough_conductor.pdf(sframe, uv);
        break;
    case MaterialType::Dielectric:
        pdf = DielectricMaterial::pdf();
        break;
    default:
        panic();
    }

    assert(pdf >= 0.f);

    return pdf;
}

spectral
Material::eval(const ShadingFrame &sframe, const SampledLambdas &lambdas,
               const vec2 &uv) const {
    if (sframe.is_degenerate()) {
        return spectral::ZERO();
    }

    spectral result{};

    switch (type) {
    case MaterialType::Diffuse:
        result = diffuse.eval(lambdas, uv);
        break;
    case MaterialType::DiffuseTransmission:
        result = diffuse_transmission.eval(lambdas, uv);
        break;
    case MaterialType::CoatedDiffuse:
        result = coated_diffuse.eval(sframe, lambdas, uv);
        break;
    case MaterialType::RoughCoatedDiffuse:
        result = rough_coated_diffuse.eval(sframe, lambdas, uv);
        break;
    case MaterialType::RoughConductor:
        result = rough_conductor.eval(sframe, lambdas, uv);
        break;
    case MaterialType::Conductor:
        result = conductor.eval(sframe, lambdas, uv);
        break;
    case MaterialType::Dielectric:
        result = DielectricMaterial::eval();
        break;
    default:
        panic();
    }

    assert(!result.is_invalid());

    return result;
}

bool
Material::is_delta() const {
    switch (type) {
    case MaterialType::Diffuse:
        return false;
    case MaterialType::DiffuseTransmission:
        return false;
    case MaterialType::CoatedDiffuse:
        return CoatedDifuseMaterial::is_dirac_delta();
    case MaterialType::RoughCoatedDiffuse:
        return false;
    case MaterialType::Conductor:
        return true;
    case MaterialType::RoughConductor:
        return false;
    case MaterialType::Dielectric:
        return true;
    default:
        panic();
    }
}
