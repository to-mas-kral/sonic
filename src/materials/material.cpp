#include "material.h"

#include "../integrator/shading_frame.h"
#include "diffuse_transmission.h"

Material
Material::make_diffuse(SpectrumTexture *reflectance) {
    return Material{.type = MaterialType::Diffuse,
                    .diffuse = {
                        .reflectance = reflectance,
                    }};
}

Material
Material::make_diffuse_transmission(SpectrumTexture *reflectance,
                                    SpectrumTexture *transmittance, f32 scale,
                                    ChunkAllocator<> &material_allocator) {
    auto *diffusetransmission_mat =
        material_allocator.allocate<DiffuseTransmissionMaterial>();

    *diffusetransmission_mat = DiffuseTransmissionMaterial{
        .reflectance = reflectance,
        .transmittace = transmittance,
        .scale = scale,
    };

    return Material{.type = MaterialType::DiffuseTransmission,
                    .diffusetransmission = diffusetransmission_mat};
}

Material
Material::make_dielectric(Spectrum ext_ior, SpectrumTexture *int_ior,
                          Spectrum transmittance, ChunkAllocator<> &material_allocator) {
    auto *dielectric_mat = material_allocator.allocate<DielectricMaterial>();
    *dielectric_mat = DielectricMaterial{
        .m_int_ior = int_ior,
        .m_ext_ior = ext_ior,
        .m_transmittance = transmittance,
    };

    return Material{.type = MaterialType::Dielectric, .dielectric = dielectric_mat};
}

Material
Material::make_conductor(SpectrumTexture *eta, SpectrumTexture *k,
                         ChunkAllocator<> &material_allocator) {
    auto *conductor_mat = material_allocator.allocate<ConductorMaterial>();
    *conductor_mat = ConductorMaterial{
        .m_perfect = false,
        .m_eta = eta,
        .m_k = k,
    };

    return Material{.type = MaterialType::Conductor, .conductor = conductor_mat};
}

Material
Material::make_rough_conductor(FloatTexture *alpha, SpectrumTexture *eta,
                               SpectrumTexture *k, ChunkAllocator<> &material_allocator) {
    auto *rough_conductor_mat = material_allocator.allocate<RoughConductorMaterial>();
    *rough_conductor_mat = RoughConductorMaterial{
        .m_eta = eta,
        .m_k = k,
        .m_alpha = alpha,
    };

    return Material{.type = MaterialType::RoughConductor,
                    .rough_conductor = rough_conductor_mat};
}

Material
Material::make_plastic(Spectrum ext_ior, Spectrum int_ior,
                       SpectrumTexture *diffuse_reflectance,
                       ChunkAllocator<> &material_allocator) {
    auto *plastic_mat = material_allocator.allocate<CoatedDifuseMaterial>();
    *plastic_mat = CoatedDifuseMaterial{
        .ext_ior = ext_ior,
        .int_ior = int_ior,
        .diffuse_reflectance = diffuse_reflectance,
    };

    return Material{.type = MaterialType::CoatedDiffuse, .coateddiffuse = plastic_mat};
}

Material
Material::make_rough_plastic(FloatTexture *alpha, Spectrum ext_ior, Spectrum int_ior,
                             SpectrumTexture *diffuse_reflectance,
                             ChunkAllocator<> &material_allocator) {
    auto *rough_plastic_mat = material_allocator.allocate<RoughCoatedDiffuseMaterial>();
    *rough_plastic_mat = RoughCoatedDiffuseMaterial{
        .m_alpha = alpha,
        .ext_ior = ext_ior,
        .int_ior = int_ior,
        .diffuse_reflectance = diffuse_reflectance,
    };

    return Material{.type = MaterialType::RoughCoatedDiffuse,
                    .rough_coateddiffuse = rough_plastic_mat};
}

std::optional<BSDFSample>
Material::sample(const ShadingFrameIncomplete &sframe, norm_vec3 wo, const vec3 &xi,
                 SampledLambdas &lambdas, const vec2 &uv,
                 const bool is_frontfacing) const {
    wo = sframe.to_local(wo).normalized();

    if (sframe.cos_theta(wo) == 0.f) {
        return {};
    }

    std::optional<BSDFSample> bsdf_sample{};

    switch (type) {
    case MaterialType::Diffuse:
        bsdf_sample = diffuse.sample(sframe, wo, vec2(xi.x, xi.y), lambdas, uv);
        break;
    case MaterialType::DiffuseTransmission:
        bsdf_sample = diffusetransmission->sample(sframe, wo, vec2(xi.x, xi.y),
                                                  lambdas, uv);
        break;
    case MaterialType::CoatedDiffuse:
        bsdf_sample = coateddiffuse->sample(sframe, wo, xi, lambdas, uv);
        break;
    case MaterialType::RoughCoatedDiffuse:
        bsdf_sample = rough_coateddiffuse->sample(sframe, wo, xi, lambdas, uv);
        break;
    case MaterialType::Conductor:
        bsdf_sample = conductor->sample(sframe, wo, lambdas, uv);
        break;
    case MaterialType::RoughConductor:
        bsdf_sample =
            rough_conductor->sample(sframe, wo, vec2(xi.x, xi.y), lambdas, uv);
        break;
    case MaterialType::Dielectric:
        bsdf_sample = dielectric->sample(sframe, wo, vec2(xi.x, xi.y), lambdas,
                                         uv, is_frontfacing);
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
        pdf = coateddiffuse->pdf(sframe, lambdas);
        break;
    case MaterialType::RoughCoatedDiffuse:
        pdf = rough_coateddiffuse->pdf(sframe, lambdas, uv);
        break;
    case MaterialType::Conductor:
        pdf = ConductorMaterial::pdf();
        break;
    case MaterialType::RoughConductor:
        pdf = rough_conductor->pdf(sframe, uv);
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
        result = diffusetransmission->eval(lambdas, uv);
        break;
    case MaterialType::CoatedDiffuse:
        result = coateddiffuse->eval(sframe, lambdas, uv);
        break;
    case MaterialType::RoughCoatedDiffuse:
        result = rough_coateddiffuse->eval(sframe, lambdas, uv);
        break;
    case MaterialType::RoughConductor:
        result = rough_conductor->eval(sframe, lambdas, uv);
        break;
    case MaterialType::Conductor:
        result = conductor->eval(sframe, lambdas, uv);
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
Material::is_dirac_delta() const {
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
