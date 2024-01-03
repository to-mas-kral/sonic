#ifndef PT_MATERIAL_H
#define PT_MATERIAL_H

#include "color/sampled_spectrum.h"
#include "color/spectrum.h"
#include "integrator/utils.h"
#include "math/sampling.h"
#include "math/vecmath.h"
#include "utils/basic_types.h"

enum class MaterialType : u8 {
    Diffuse = 0,
    Conductor = 1,
    Dielectric = 2,
};

struct BSDFSample {
    spectral bsdf;
    norm_vec3 wi;
    f32 pdf;
    bool did_refract = false;
};

struct Material {
    static Material
    make_diffuse(const RgbSpectrum &p_reflectance) {
        return Material{.type = MaterialType::Diffuse,
                        .diffuse = {
                            .reflectance = p_reflectance,
                            .reflectance_tex_id = {},
                        }};
    }

    static Material
    make_diffuse(u32 reflectance_tex_id) {
        return Material{.type = MaterialType::Diffuse,
                        .diffuse = {
                            .reflectance = RgbSpectrum{},
                            .reflectance_tex_id = reflectance_tex_id,
                        }};
    }

    static Material
    make_dielectric(Spectrum ext_ior, Spectrum int_ior, Spectrum transmittance) {
        return Material{.type = MaterialType::Dielectric,
                        .dielectric = {
                            .int_ior = int_ior,
                            .ext_ior = ext_ior,
                            .transmittance = transmittance,
                        }};
    }

    static Material
    make_conductor() {
        return Material{
            .type = MaterialType::Conductor,
        };
    }

    __device__ BSDFSample
    sample_diffuse(const norm_vec3 &normal, const norm_vec3 &wo, const vec2 &sample,
                   const SampledLambdas &lambdas, const Texture *textures,
                   const vec2 &uv) const {
        norm_vec3 sample_dir = sample_cosine_hemisphere(sample);
        norm_vec3 wi = orient_dir(sample_dir, normal);
        auto sgeom = get_shading_geom(normal, wi, wo);
        return BSDFSample{
            .bsdf = eval(sgeom, lambdas, textures, uv),
            .wi = wi,
            .pdf = pdf(sgeom),
            .did_refract = false,
        };
    }

    __device__ BSDFSample
    sample_conductor(const norm_vec3 &normal, const norm_vec3 &wo,
                     const SampledLambdas &lambdas, const Texture *textures,
                     const vec2 &uv) const {
        norm_vec3 wi = vec3::reflect(wo, normal).normalized();
        auto sgeom = get_shading_geom(normal, wi, wo);
        return BSDFSample{
            .bsdf = eval(sgeom, lambdas, textures, uv),
            .wi = wi,
            .pdf = 1.f,
            .did_refract = false,
        };
    }

    /// Taken from PBRTv4
    __device__ static COption<vec3>
    refract(const norm_vec3 &wo, const norm_vec3 &normal, f32 rel_ior) {
        f32 cos_theta_i = vec3::dot(normal, wo);
        f32 sin2_theta_i = max(0.f, 1.f - sqr(cos_theta_i));
        f32 sin2_theta_t = sin2_theta_i / sqr(rel_ior);

        // Total internal reflection
        if (sin2_theta_t >= 1.f) {
            return {};
        }

        f32 cos_theta_t = sqrt(1.f - sin2_theta_t);
        return (-wo / rel_ior) + normal * ((cos_theta_i / rel_ior) - cos_theta_t);
    }

    /// Taken from PBRTv4
    __device__ static f32
    fresnel_dielectric(f32 rel_ior, f32 cos_theta_i) {
        f32 sin2_theta_i = 1.f - sqr(cos_theta_i);
        f32 sin2_theta_t = sin2_theta_i / sqr(rel_ior);
        if (sin2_theta_t >= 1.f) {
            // Total internal reflection
            return 1.f;
        } else {
            f32 cos_theta_t = sqrt(1.f - sin2_theta_t);

            f32 r_parl = (rel_ior * cos_theta_i - cos_theta_t) /
                         (rel_ior * cos_theta_i + cos_theta_t);
            f32 r_perp = (cos_theta_i - rel_ior * cos_theta_t) /
                         (cos_theta_i + rel_ior * cos_theta_t);
            return (sqr(r_parl) + sqr(r_perp)) / 2.f;
        }
    }
    __device__ BSDFSample
    sample_dielectric(const norm_vec3 &normal, const norm_vec3 &wo, const vec2 &sample,
                      const SampledLambdas &lambdas, const Texture *textures,
                      const vec2 &uv, bool is_frontfacing) const {
        f32 int_ior = dielectric.int_ior.eval_single(lambdas[0]);
        f32 ext_ior = dielectric.ext_ior.eval_single(lambdas[0]);
        f32 rel_ior = int_ior / ext_ior;
        if (!is_frontfacing) {
            rel_ior = 1.f / rel_ior;
        }

        f32 cos_theta_i = vec3::dot(wo, normal);
        f32 fresnel_refl = fresnel_dielectric(rel_ior, cos_theta_i);

        auto reflect = [&]() {
            auto wi = vec3::reflect(wo, normal).normalized();
            auto sgeom = get_shading_geom(normal, wi, wo);

            return BSDFSample{
                .bsdf = spectral::make_constant(fresnel_refl) / sgeom.cos_theta,
                .wi = wi,
                .pdf = fresnel_refl,
                .did_refract = false,
            };
        };

        if (sample.x < fresnel_refl) {
            return reflect();
        } else {
            auto refr = refract(wo, normal, rel_ior);
            if (refr.has_value()) {
                auto wi = refr.value().normalized();
                auto sgeom = get_shading_geom(normal, wi, wo);
                f32 transmittance = dielectric.transmittance.eval_single(lambdas[0]);

                return BSDFSample{
                    .bsdf = spectral::make_constant(1.f - fresnel_refl) * transmittance /
                            sgeom.cos_theta,
                    .wi = wi,
                    .pdf = 1.f - fresnel_refl,
                    .did_refract = true,
                };
            } else {
                // Total internal reflection
                return reflect();
            }
        }
    }

    __device__ BSDFSample
    sample(const norm_vec3 &normal, const norm_vec3 &wo, const vec2 &sample,
           const SampledLambdas &lambdas, const Texture *textures, const vec2 &uv,
           bool is_frontfacing) const {
        switch (type) {
        case MaterialType::Diffuse: {
            return sample_diffuse(normal, wo, sample, lambdas, textures, uv);
        }
        case MaterialType::Conductor: {
            return sample_conductor(normal, wo, lambdas, textures, uv);
        }
        case MaterialType::Dielectric:
            return sample_dielectric(normal, wo, sample, lambdas, textures, uv,
                                     is_frontfacing);
        }
    }

    // Probability density function of sampling the BRDF
    __device__ f32
    pdf(const ShadingGeometry &sgeom) const {
        switch (type) {
        case MaterialType::Diffuse: {
            return sgeom.cos_theta / M_PIf;
        }
        case MaterialType::Conductor: {
            return 0.f;
        }
        case MaterialType::Dielectric:
            return 0.f;
        }
    }

    __device__ spectral
    eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
         const Texture *textures, const vec2 &uv) const {
        switch (type) {
        case MaterialType::Diffuse: {
            spectral refl{};

            if (diffuse.reflectance_tex_id.has_value()) {
                auto tex_id = diffuse.reflectance_tex_id.value();
                auto texture = &textures[tex_id];

                auto sigmoid_coeff = texture->fetch(uv);
                refl = RgbSpectrum::from_coeff(sigmoid_coeff).eval(lambdas);
            } else {
                refl = diffuse.reflectance.eval(lambdas);
            }

            return refl / M_PIf;
        }
        case MaterialType::Conductor:
            return spectral::ONE() / sgeom.cos_theta;
        case MaterialType::Dielectric:
            assert(false);
            return spectral::make_constant(NAN);
        }
    }

    __device__ bool
    is_specular() const {
        switch (type) {
        case MaterialType::Diffuse:
            return false;
        case MaterialType::Conductor:
            return true;
        case MaterialType::Dielectric:
            return true;
        }
    }

    // TODO: discriminated ptr would be nice here...
    MaterialType type = MaterialType::Diffuse;
    bool is_twosided = false;
    union {
        struct {
            // TODO: merge into some Constant texture type...
            RgbSpectrum reflectance;
            COption<u32> reflectance_tex_id = {};
        } diffuse;
        struct {
            Spectrum int_ior;
            Spectrum ext_ior;
            Spectrum transmittance;
        } dielectric;
    };
};

#endif // PT_MATERIAL_H
