#include "material.h"

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
    auto *plastic_mat = material_allocator.allocate<PlasticMaterial>();
    *plastic_mat = PlasticMaterial{
        .ext_ior = ext_ior,
        .int_ior = int_ior,
        .diffuse_reflectance = diffuse_reflectance,
    };

    return Material{.type = MaterialType::Plastic, .plastic = plastic_mat};
}

Material
Material::make_rough_plastic(FloatTexture *alpha, Spectrum ext_ior, Spectrum int_ior,
                             SpectrumTexture *diffuse_reflectance,
                             ChunkAllocator<> &material_allocator) {
    auto *rough_plastic_mat = material_allocator.allocate<RoughPlasticMaterial>();
    *rough_plastic_mat = RoughPlasticMaterial{
        .m_alpha = alpha,
        .ext_ior = ext_ior,
        .int_ior = int_ior,
        .diffuse_reflectance = diffuse_reflectance,
    };

    return Material{.type = MaterialType::RoughPlastic,
                    .rough_plastic = rough_plastic_mat};
}

Option<BSDFSample>
Material::sample(const norm_vec3 &normal, const norm_vec3 &wo, const vec3 &sample,
                 SampledLambdas &lambdas, const vec2 &uv, bool is_frontfacing) const {
    auto nowo = vec3::dot(normal, wo);
    if (nowo == 0.f) {
        return {};
    }

    Option<BSDFSample> bsdf_sample{};

    switch (type) {
    case MaterialType::Diffuse:
        bsdf_sample = diffuse.sample(normal, wo, vec2(sample.x, sample.y), lambdas, uv);
        break;
    case MaterialType::DiffuseTransmission:
        bsdf_sample = diffusetransmission->sample(normal, wo, vec2(sample.x, sample.y),
                                                  lambdas, uv);
        break;
    case MaterialType::Plastic:
        bsdf_sample = plastic->sample(normal, wo, sample, lambdas, uv);
        break;
    case MaterialType::RoughPlastic:
        bsdf_sample = rough_plastic->sample(normal, wo, sample, lambdas, uv);
        break;
    case MaterialType::Conductor:
        bsdf_sample = conductor->sample(normal, wo, lambdas, uv);
        break;
    case MaterialType::RoughConductor:
        bsdf_sample =
            rough_conductor->sample(normal, wo, vec2(sample.x, sample.y), lambdas, uv);
        break;
    case MaterialType::Dielectric:
        bsdf_sample = dielectric->sample(normal, wo, vec2(sample.x, sample.y), lambdas,
                                         uv, is_frontfacing);
        break;
    default:
        assert(false);
    }

    if (bsdf_sample.has_value()) {
        assert(!bsdf_sample->bsdf.is_invalid());
    }

    return bsdf_sample;
}

f32
Material::pdf(const ShadingGeometry &sgeom, const SampledLambdas &λ,
              const vec2 &uv) const {
    f32 pdf{};

    switch (type) {
    case MaterialType::Diffuse:
        pdf = DiffuseMaterial::pdf(sgeom);
        break;
    case MaterialType::DiffuseTransmission:
        pdf = DiffuseTransmissionMaterial::pdf(sgeom);
        break;
    case MaterialType::Plastic:
        pdf = plastic->pdf(sgeom, λ);
        break;
    case MaterialType::RoughPlastic:
        pdf = rough_plastic->pdf(sgeom, λ, uv);
        break;
    case MaterialType::Conductor:
        pdf = ConductorMaterial::pdf();
        break;
    case MaterialType::RoughConductor:
        pdf = rough_conductor->pdf(sgeom, uv);
        break;
    case MaterialType::Dielectric:
        pdf = DielectricMaterial::pdf();
        break;
    default:
        assert(false);
    }

    assert(pdf >= 0.f);

    return pdf;
}

spectral
Material::eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
               const vec2 &uv) const {
    if (sgeom.is_degenerate()) {
        return spectral::ZERO();
    }

    spectral result{};

    switch (type) {
    case MaterialType::Diffuse:
        result = diffuse.eval(sgeom, lambdas, uv);
        break;
    case MaterialType::DiffuseTransmission:
        result = diffusetransmission->eval(sgeom, lambdas, uv);
        break;
    case MaterialType::Plastic:
        result = plastic->eval(sgeom, lambdas, uv);
        break;
    case MaterialType::RoughPlastic:
        result = rough_plastic->eval(sgeom, lambdas, uv);
        break;
    case MaterialType::RoughConductor:
        result = rough_conductor->eval(sgeom, lambdas, uv);
        break;
    case MaterialType::Conductor:
        result = conductor->eval(sgeom, lambdas, uv);
        break;
    case MaterialType::Dielectric:
        result = DielectricMaterial::eval();
        break;
    default:
        assert(false);
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
    default:
        assert(false);
    }
}
