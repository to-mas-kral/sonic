#ifndef PT_PLASTIC_H
#define PT_PLASTIC_H

#include "../color/spectrum.h"
#include "../integrator/utils.h"
#include "../math/sampling.h"
#include "../utils/basic_types.h"
#include "bsdf_sample.h"
#include "common.h"

struct PlasticMaterial {
    __host__ __device__ f32
    pdf(const ShadingGeometry &sgeom, const SampledLambdas &λ) const {
        f32 int_η = int_ior.eval_single(λ[0]);
        f32 ext_η = ext_ior.eval_single(λ[0]);
        f32 rel_η = int_η / ext_η;

        f32 fresnel_i = fresnel_dielectric(rel_η, sgeom.nowo);

        return (1.f - fresnel_i) * sgeom.cos_theta / M_PIf;
    }

    __host__ __device__ static bool
    is_dirac_delta() {
        return true;
    }
    /// Math taken from:
    /// Physically Based Specular + Diffuse
    /// Jan van Bergen
    __device__ spectral
    eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas) const {
        f32 int_ior_s = int_ior.eval_single(lambdas[0]);
        f32 ext_ior_s = ext_ior.eval_single(lambdas[0]);
        /// This is external / internal !
        f32 rel_ior = ext_ior_s / int_ior_s;

        f32 fresnel_i = fresnel_dielectric(1.f / rel_ior, sgeom.nowi);

        if (sgeom.noh > 0.99999f) {
            // Specular case
            return spectral::make_constant(fresnel_i);
        } else {
            // Figure out the angle of the ray that refracts from inside to outisde ωo
            f32 cos_theta_out = sgeom.nowo;
            f32 sin_theta_out = sqrt(1.f - sqr(cos_theta_out));
            f32 sin_theta_in = sin_theta_out * rel_ior;
            f32 cos_theta_in = sqrt(1.f - sqr(sin_theta_in));
            f32 fresnel_o = fresnel_dielectric(rel_ior, cos_theta_in);

            spectral α = diffuse_reflectance.eval(lambdas);

            // clang-format off
            f32 re = 0.919317f
                 - (3.4793f / int_ior_s)
                 + (6.75335f / cuda::std::pow(int_ior_s, 2))
                 - (7.80989f / cuda::std::pow(int_ior_s, 3))
                 + (4.98554f / cuda::std::pow(int_ior_s, 4))
                 - (1.36881f / cuda::std::pow(int_ior_s, 5));
            // clang-format on

            f32 ri = 1.f - sqr(rel_ior) * (1.f - re);

            spectral scattering_brdf = α * (1.f - fresnel_i) * (1.f - fresnel_o) /
                                       ((spectral::make_constant(1.f) - α * ri) * M_PIf);

            spectral brdf = scattering_brdf * sqr(rel_ior);

            return brdf;
        }
    }

    __device__ BSDFSample
    sample(const norm_vec3 &normal, const norm_vec3 &ωo, const vec3 &ξ,
           const SampledLambdas &λ) const {
        f32 int_η = int_ior.eval_single(λ[0]);
        f32 ext_η = ext_ior.eval_single(λ[0]);
        f32 rel_η = int_η / ext_η;

        f32 noωo = vec3::dot(normal, ωo);
        f32 potential_fresnel = fresnel_dielectric(rel_η, noωo);

        if (ξ.z < potential_fresnel) {
            // sample specular
            norm_vec3 ωi = vec3::reflect(ωo, normal).normalized();
            auto sgeom = ShadingGeometry::make(normal, ωi, ωo);

            f32 specular_brdf = potential_fresnel / sgeom.cos_theta;

            return BSDFSample{
                .bsdf = spectral::make_constant(specular_brdf),
                .wi = ωi,
                .pdf = potential_fresnel,
                .did_refract = false,
            };
        } else {
            // sample diffuse
            norm_vec3 sample_dir = sample_cosine_hemisphere(vec2(ξ.x, ξ.y));
            norm_vec3 ωi = orient_dir(sample_dir, normal);
            auto sgeom = ShadingGeometry::make(normal, ωi, ωo);

            return BSDFSample{
                .bsdf = eval(sgeom, λ),
                .wi = ωi,
                .pdf = pdf(sgeom, λ),
                .did_refract = false,
            };
        }
    }

    Spectrum ext_ior;
    Spectrum int_ior;
    Spectrum diffuse_reflectance;
};

#endif // PT_PLASTIC_H
