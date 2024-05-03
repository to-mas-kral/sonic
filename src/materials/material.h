#ifndef PT_MATERIAL_H
#define PT_MATERIAL_H

#include "../color/sampled_spectrum.h"
#include "../color/spectrum.h"
#include "../integrator/utils.h"
#include "../scene/texture.h"
#include "../scene/texture_id.h"
#include "../utils/basic_types.h"
#include "bsdf_sample.h"
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
    make_diffuse(TextureId reflectance_tex_id);

    static Material
    make_dielectric(Spectrum ext_ior, TextureId int_ior, Spectrum transmittance,
                    ChunkAllocator<> &material_allocator);

    static Material
    make_conductor(TextureId eta, TextureId k, ChunkAllocator<> &material_allocator);

    static Material
    make_conductor_perfect(ChunkAllocator<> &material_allocator);

    static Material
    make_rough_conductor(TextureId alpha, TextureId eta, TextureId k,
                         ChunkAllocator<> &material_allocator);

    static Material
    make_plastic(Spectrum ext_ior, Spectrum int_ior, TextureId diffuse_reflectance_id,
                 ChunkAllocator<> &material_allocator);

    static Material
    make_rough_plastic(TextureId alpha, Spectrum ext_ior, Spectrum int_ior,
                       TextureId diffuse_reflectance_id,
                       ChunkAllocator<> &material_allocator);

    // TODO: change Texture* to std::span<>
    Option<BSDFSample>
    sample(const norm_vec3 &normal, const norm_vec3 &wo, const vec3 &sample,
           const SampledLambdas &lambdas, const Texture *textures, const vec2 &uv,
           bool is_frontfacing) const;

    // Probability density function of sampling the BRDF
    f32
    pdf(const ShadingGeometry &sgeom, const SampledLambdas &λ, const Texture *textures,
        const vec2 &uv) const;

    spectral
    eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
         const Texture *textures, const vec2 &uv) const;

    bool
    is_dirac_delta() const;

    MaterialType type = MaterialType::Diffuse;
    bool is_twosided = false;

    union {
        DiffuseMaterial diffuse;
        PlasticMaterial *plastic;
        RoughPlasticMaterial *rough_plastic;
        DielectricMaterial *dielectric;
        ConductorMaterial *conductor;
        RoughConductorMaterial *rough_conductor;
    };
};

#endif // PT_MATERIAL_H
