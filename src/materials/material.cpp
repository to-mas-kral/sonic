#include "material.h"

#include "../integrator/shading_frame.h"
#include "diffuse_transmission.h"

Material
Material::make_diffuse(SpectrumTexture *reflectance) {
    return Material{.type = MaterialType::Diffuse,
                    .diffuse = DiffuseMaterial(reflectance)};
}

Material
Material::make_diffuse_transmission(SpectrumTexture *reflectance,
                                    SpectrumTexture *transmittance, const f32 scale) {
    const auto diffusetransmission_mat =
        DiffuseTransmissionMaterial(reflectance, transmittance, scale);

    return Material{.type = MaterialType::DiffuseTransmission,
                    .diffusetransmission = diffusetransmission_mat};
}

Material
Material::make_dielectric(const Spectrum &ext_ior, SpectrumTexture *int_ior,
                          const Spectrum &transmittance) {
    const auto dielectric_mat = DielectricMaterial(int_ior, ext_ior, transmittance);
    return Material{.type = MaterialType::Dielectric, .dielectric = dielectric_mat};
}

Material
Material::make_conductor(SpectrumTexture *eta, SpectrumTexture *k) {
    const auto conductor_mat = ConductorMaterial(false, eta, k);
    return Material{.type = MaterialType::Conductor, .conductor = conductor_mat};
}

Material
Material::make_rough_conductor(FloatTexture *alpha, SpectrumTexture *eta,
                               SpectrumTexture *k) {
    const auto rough_conductor_material = RoughConductorMaterial(eta, k, alpha);
    return Material{.type = MaterialType::RoughConductor,
                    .rough_conductor = rough_conductor_material};
}

Material
Material::make_plastic(const Spectrum &ext_ior, const Spectrum &int_ior,
                       SpectrumTexture *diffuse_reflectance) {
    const auto plastic_mat = CoatedDifuseMaterial(ext_ior, int_ior, diffuse_reflectance);
    return Material{.type = MaterialType::CoatedDiffuse, .coateddiffuse = plastic_mat};
}

Material
Material::make_rough_plastic(FloatTexture *alpha, const Spectrum &ext_ior,
                             const Spectrum &int_ior,
                             SpectrumTexture *diffuse_reflectance) {
    const auto rough_plastic_mat =
        RoughCoatedDiffuseMaterial(alpha, ext_ior, int_ior, diffuse_reflectance);
    return Material{.type = MaterialType::RoughCoatedDiffuse,
                    .rough_coateddiffuse = rough_plastic_mat};
}

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
            diffusetransmission.sample(sframe, wo, vec2(xi.x, xi.y), lambdas, uv);
        break;
    case MaterialType::CoatedDiffuse:
        bsdf_sample = coateddiffuse.sample(sframe, wo, xi, lambdas, uv);
        break;
    case MaterialType::RoughCoatedDiffuse:
        bsdf_sample = rough_coateddiffuse.sample(sframe, wo, xi, lambdas, uv);
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
        pdf = coateddiffuse.pdf(sframe, lambdas);
        break;
    case MaterialType::RoughCoatedDiffuse:
        pdf = rough_coateddiffuse.pdf(sframe, lambdas, uv);
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
        result = diffusetransmission.eval(lambdas, uv);
        break;
    case MaterialType::CoatedDiffuse:
        result = coateddiffuse.eval(sframe, lambdas, uv);
        break;
    case MaterialType::RoughCoatedDiffuse:
        result = rough_coateddiffuse.eval(sframe, lambdas, uv);
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
