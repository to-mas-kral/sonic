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
                 const SampledLambdas &lambdas, const vec2 &uv,
                 bool is_frontfacing) const {
    auto nowo = vec3::dot(normal, wo);
    if (nowo == 0.f) {
        return {};
    }

    switch (type) {
    case MaterialType::Diffuse:
        return diffuse.sample(normal, wo, vec2(sample.x, sample.y), lambdas, uv);
    case MaterialType::DiffuseTransmission:
        return diffusetransmission->sample(normal, wo, vec2(sample.x, sample.y), lambdas,
                                           uv);
    case MaterialType::Plastic:
        return plastic->sample(normal, wo, sample, lambdas, uv);
    case MaterialType::RoughPlastic:
        return rough_plastic->sample(normal, wo, sample, lambdas, uv);
    case MaterialType::Conductor:
        return conductor->sample(normal, wo, lambdas, uv);
    case MaterialType::RoughConductor:
        return rough_conductor->sample(normal, wo, vec2(sample.x, sample.y), lambdas, uv);
    case MaterialType::Dielectric:
        return dielectric->sample(normal, wo, vec2(sample.x, sample.y), lambdas, uv,
                                  is_frontfacing);
    default:
        assert(false);
    }
}

f32
Material::pdf(const ShadingGeometry &sgeom, const SampledLambdas &λ,
              const vec2 &uv) const {
    switch (type) {
    case MaterialType::Diffuse:
        return DiffuseMaterial::pdf(sgeom);
    case MaterialType::DiffuseTransmission:
        return DiffuseTransmissionMaterial::pdf(sgeom);
    case MaterialType::Plastic:
        return plastic->pdf(sgeom, λ);
    case MaterialType::RoughPlastic:
        return rough_plastic->pdf(sgeom, λ, uv);
    case MaterialType::Conductor:
        return ConductorMaterial::pdf();
    case MaterialType::RoughConductor:
        return rough_conductor->pdf(sgeom, uv);
    case MaterialType::Dielectric:
        return DielectricMaterial::pdf();
    default:
        assert(false);
    }
}

spectral
Material::eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
               const vec2 &uv) const {
    if (sgeom.is_degenerate()) {
        return spectral::ZERO();
    }

    switch (type) {
    case MaterialType::Diffuse:
        return diffuse.eval(sgeom, lambdas, uv);
    case MaterialType::DiffuseTransmission:
        return diffusetransmission->eval(sgeom, lambdas, uv);
    case MaterialType::Plastic:
        return plastic->eval(sgeom, lambdas, uv);
    case MaterialType::RoughPlastic:
        return rough_plastic->eval(sgeom, lambdas, uv);
    case MaterialType::RoughConductor:
        return rough_conductor->eval(sgeom, lambdas, uv);
    case MaterialType::Conductor:
        return conductor->eval(sgeom, lambdas, uv);
    case MaterialType::Dielectric:
        return DielectricMaterial::eval();
    default:
        assert(false);
    }
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
