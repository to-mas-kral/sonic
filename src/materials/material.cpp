#include "material.h"

Material
Material::make_diffuse(u32 reflectance_tex_id) {
    return Material{.type = MaterialType::Diffuse,
                    .diffuse = {
                        .reflectance_tex_id = reflectance_tex_id,
                    }};
}

Material
Material::make_dielectric(Spectrum ext_ior, Spectrum int_ior, Spectrum transmittance,
                          ChunkAllocator<> &material_allocator) {
    auto *dielectric_mat = material_allocator.allocate<DielectricMaterial>();
    *dielectric_mat = DielectricMaterial{
        .m_int_ior = int_ior,
        .m_ext_ior = ext_ior,
        .m_transmittance = transmittance,
    };

    return Material{.type = MaterialType::Dielectric, .dielectric = dielectric_mat};
}

Material
Material::make_conductor(Spectrum eta, Spectrum k, ChunkAllocator<> &material_allocator) {
    auto *conductor_mat = material_allocator.allocate<ConductorMaterial>();
    *conductor_mat = ConductorMaterial{
        .m_perfect = false,
        .m_eta = eta,
        .m_k = k,
    };

    return Material{.type = MaterialType::Conductor, .conductor = conductor_mat};
}

Material
Material::make_conductor_perfect(ChunkAllocator<> &material_allocator) {
    auto *conductor_mat = material_allocator.allocate<ConductorMaterial>();
    *conductor_mat = ConductorMaterial{
        .m_perfect = true,
        .m_eta = Spectrum(ConstantSpectrum::make(0.f)),
        .m_k = Spectrum(ConstantSpectrum::make(0.f)),
    };

    return Material{.type = MaterialType::Conductor, .conductor = conductor_mat};
}

Material
Material::make_rough_conductor(f32 alpha, Spectrum eta, Spectrum k,
                               ChunkAllocator<> &material_allocator) {
    if (TrowbridgeReitzGGX::is_alpha_effectively_zero(alpha)) {
        auto *conductor_mat = material_allocator.allocate<ConductorMaterial>();
        *conductor_mat = ConductorMaterial{
            .m_perfect = false,
            .m_eta = eta,
            .m_k = k,
        };

        return Material{.type = MaterialType::Conductor, .conductor = conductor_mat};
    } else {
        auto *rough_conductor_mat = material_allocator.allocate<RoughConductorMaterial>();
        *rough_conductor_mat = RoughConductorMaterial{
            .m_eta = eta,
            .m_k = k,
            .m_alpha = alpha,
        };

        return Material{.type = MaterialType::RoughConductor,
                        .rough_conductor = rough_conductor_mat};
    }
}

Material
Material::make_plastic(Spectrum ext_ior, Spectrum int_ior, u32 diffuse_reflectance_id,
                       ChunkAllocator<> &material_allocator) {
    auto *plastic_mat = material_allocator.allocate<PlasticMaterial>();
    *plastic_mat = PlasticMaterial{
        .ext_ior = ext_ior,
        .int_ior = int_ior,
        .diffuse_reflectance_id = diffuse_reflectance_id,
    };

    return Material{.type = MaterialType::Plastic, .plastic = plastic_mat};
}

Material
Material::make_rough_plastic(f32 alpha, Spectrum ext_ior, Spectrum int_ior,
                             u32 diffuse_reflectance_id,
                             ChunkAllocator<> &material_allocator) {
    if (TrowbridgeReitzGGX::is_alpha_effectively_zero(alpha)) {
        auto *plastic_mat = material_allocator.allocate<PlasticMaterial>();
        *plastic_mat = PlasticMaterial{
            .ext_ior = ext_ior,
            .int_ior = int_ior,
            .diffuse_reflectance_id = diffuse_reflectance_id,
        };

        return Material{.type = MaterialType::Plastic, .plastic = plastic_mat};
    } else {
        auto *rough_plastic_mat = material_allocator.allocate<RoughPlasticMaterial>();
        *rough_plastic_mat = RoughPlasticMaterial{
            .alpha = alpha,
            .ext_ior = ext_ior,
            .int_ior = int_ior,
            .diffuse_reflectance_id = diffuse_reflectance_id,
        };

        return Material{.type = MaterialType::RoughPlastic,
                        .rough_plastic = rough_plastic_mat};
    }
}

Option<BSDFSample>
Material::sample(const norm_vec3 &normal, const norm_vec3 &wo, const vec3 &sample,
                 const SampledLambdas &lambdas, const Texture *textures, const vec2 &uv,
                 bool is_frontfacing) const {
    switch (type) {
    case MaterialType::Diffuse:
        return diffuse.sample(normal, wo, vec2(sample.x, sample.y), lambdas, textures,
                              uv);
    case MaterialType::Plastic:
        return plastic->sample(normal, wo, sample, lambdas, textures, uv);
    case MaterialType::RoughPlastic:
        return rough_plastic->sample(normal, wo, sample, lambdas, textures, uv);
    case MaterialType::Conductor:
        return conductor->sample(normal, wo, lambdas, textures, uv);
    case MaterialType::RoughConductor:
        return rough_conductor->sample(normal, wo, vec2(sample.x, sample.y), lambdas,
                                       textures, uv);
    case MaterialType::Dielectric:
        return dielectric->sample(normal, wo, vec2(sample.x, sample.y), lambdas, textures,
                                  uv, is_frontfacing);
    }
}

f32
Material::pdf(const ShadingGeometry &sgeom, const SampledLambdas &λ) const {
    switch (type) {
    case MaterialType::Diffuse:
        return DiffuseMaterial::pdf(sgeom);
    case MaterialType::Plastic:
        return plastic->pdf(sgeom, λ);
    case MaterialType::RoughPlastic:
        return rough_plastic->pdf(sgeom, λ);
    case MaterialType::Conductor:
        return ConductorMaterial::pdf();
    case MaterialType::RoughConductor:
        return rough_conductor->pdf(sgeom);
    case MaterialType::Dielectric:
        return DielectricMaterial::pdf();
    }
}

spectral
Material::eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
               const Texture *textures, const vec2 &uv) const {
    switch (type) {
    case MaterialType::Diffuse:
        return diffuse.eval(sgeom, lambdas, textures, uv);
    case MaterialType::Plastic:
        return plastic->eval(sgeom, lambdas, textures, uv);
    case MaterialType::RoughPlastic:
        return rough_plastic->eval(sgeom, lambdas, textures, uv);
    case MaterialType::RoughConductor:
        return rough_conductor->eval(sgeom, lambdas);
    case MaterialType::Conductor:
        return conductor->eval(sgeom, lambdas, textures, uv);
    case MaterialType::Dielectric:
        return DielectricMaterial::eval();
    }
}

bool
Material::is_dirac_delta() const {
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
