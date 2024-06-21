#ifndef PT_MATERIAL_H
#define PT_MATERIAL_H

#include "../color/sampled_spectrum.h"
#include "../color/spectrum.h"
#include "../integrator/utils.h"
#include "../scene/texture.h"
#include "../utils/basic_types.h"
#include "bsdf_sample.h"
#include "conductor.h"
#include "dielectric.h"
#include "diffuse.h"
#include "diffuse_transmission.h"
#include "plastic.h"
#include "rough_conductor.h"
#include "rough_plastic.h"

enum class MaterialType : u8 {
    Diffuse,
    DiffuseTransmission,
    Plastic,
    RoughPlastic,
    Conductor,
    RoughConductor,
    Dielectric,
};

struct Material {
    static Material
    make_diffuse(SpectrumTexture *reflectance);

    static Material
    make_diffuse_transmission(SpectrumTexture *reflectance,
                              SpectrumTexture *transmittance, f32 scale,
                              ChunkAllocator<> &material_allocator);

    static Material
    make_dielectric(Spectrum ext_ior, SpectrumTexture *int_ior, Spectrum transmittance,
                    ChunkAllocator<> &material_allocator);

    static Material
    make_conductor(SpectrumTexture *eta, SpectrumTexture *k,
                   ChunkAllocator<> &material_allocator);

    static Material
    make_rough_conductor(FloatTexture *alpha, SpectrumTexture *eta, SpectrumTexture *k,
                         ChunkAllocator<> &material_allocator);

    static Material
    make_plastic(Spectrum ext_ior, Spectrum int_ior, SpectrumTexture *diffuse_reflectance,
                 ChunkAllocator<> &material_allocator);

    static Material
    make_rough_plastic(FloatTexture *alpha, Spectrum ext_ior, Spectrum int_ior,
                       SpectrumTexture *diffuse_reflectance,
                       ChunkAllocator<> &material_allocator);

    Option<BSDFSample>
    sample(const norm_vec3 &normal, const norm_vec3 &wo, const vec3 &sample,
           SampledLambdas &lambdas, const vec2 &uv, bool is_frontfacing) const;

    // Probability density function of sampling the BRDF
    f32
    pdf(const ShadingGeometry &sgeom, const SampledLambdas &λ, const vec2 &uv) const;

    spectral
    eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
         const vec2 &uv) const;

    bool
    is_dirac_delta() const;

    MaterialType type = MaterialType::Diffuse;
    bool is_twosided = false;

    union {
        DiffuseMaterial diffuse;
        DiffuseTransmissionMaterial *diffusetransmission;
        PlasticMaterial *plastic;
        RoughPlasticMaterial *rough_plastic;
        DielectricMaterial *dielectric;
        ConductorMaterial *conductor;
        RoughConductorMaterial *rough_conductor;
    };
};

#endif // PT_MATERIAL_H
