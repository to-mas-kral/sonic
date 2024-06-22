#include "plastic.h"

#include "../math/sampling.h"
#include "common.h"

f32
PlasticMaterial::pdf(const ShadingGeometry &sgeom, const SampledLambdas &λ) const {
    const f32 int_η = int_ior.eval_single(λ[0]);
    const f32 ext_η = ext_ior.eval_single(λ[0]);
    const f32 rel_η = int_η / ext_η;

    const f32 fresnel_i = fresnel_dielectric(rel_η, sgeom.nowo);

    return (1.f - fresnel_i) * sgeom.cos_theta / M_PIf;
}

bool
PlasticMaterial::is_dirac_delta() {
    return true;
}

spectral
PlasticMaterial::eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
                      const vec2 &uv) const {
    const f32 int_ior_s = int_ior.eval_single(lambdas[0]);
    const f32 ext_ior_s = ext_ior.eval_single(lambdas[0]);
    /// This is external / internal !
    const f32 rel_ior = ext_ior_s / int_ior_s;

    const f32 fresnel_i = fresnel_dielectric(1.f / rel_ior, sgeom.nowi);

    if (sgeom.noh > 0.99999f) {
        // Specular case
        return spectral::make_constant(fresnel_i);
    } else {
        // Figure out the angle of the ray that refracts from inside to outisde ωo
        const f32 cos_theta_out = sgeom.nowo;
        const f32 sin_theta_out = std::sqrt(1.f - sqr(cos_theta_out));
        const f32 sin_theta_in = sin_theta_out * rel_ior;
        const f32 cos_theta_in = std::sqrt(1.f - sqr(sin_theta_in));
        const f32 fresnel_o = fresnel_dielectric(rel_ior, cos_theta_in);

        const auto α = fetch_reflectance(diffuse_reflectance, uv, lambdas);

        f32 re = 0.919317f;
        f32 ior_pow = int_ior_s;
        re -= (3.4793f / ior_pow);
        ior_pow *= ior_pow;
        re += (6.75335f / ior_pow);
        ior_pow *= ior_pow;
        re -= (7.80989f / ior_pow);
        ior_pow *= ior_pow;
        re += (4.98554f / ior_pow);
        ior_pow *= ior_pow;
        re -= (1.36881f / ior_pow);

        const f32 ri = 1.f - sqr(rel_ior) * (1.f - re);

        const spectral scattering_brdf =
            α * (1.f - fresnel_i) * (1.f - fresnel_o) /
            ((spectral::make_constant(1.f) - α * ri) * M_PIf);

        const spectral brdf = scattering_brdf * sqr(rel_ior);

        return brdf;
    }
}

BSDFSample
PlasticMaterial::sample(const norm_vec3 &normal, const norm_vec3 &ωo, const vec3 &ξ,
                        const SampledLambdas &λ, const vec2 &uv) const {
    const f32 int_η = int_ior.eval_single(λ[0]);
    const f32 ext_η = ext_ior.eval_single(λ[0]);
    const f32 rel_η = int_η / ext_η;

    const f32 noωo = vec3::dot(normal, ωo);
    const f32 potential_fresnel = fresnel_dielectric(rel_η, noωo);

    if (ξ.z < potential_fresnel) {
        // sample specular
        const norm_vec3 ωi = vec3::reflect(ωo, normal).normalized();
        auto sgeom = ShadingGeometry::make(normal, ωi, ωo);

        const f32 specular_brdf = potential_fresnel / sgeom.cos_theta;

        return BSDFSample{
            .bsdf = spectral::make_constant(specular_brdf),
            .wi = ωi,
            .pdf = potential_fresnel,
            .did_refract = false,
        };
    } else {
        // sample diffuse
        const norm_vec3 sample_dir = sample_cosine_hemisphere(vec2(ξ.x, ξ.y));
        const norm_vec3 ωi = transform_frame(sample_dir, normal);
        const auto sgeom = ShadingGeometry::make(normal, ωi, ωo);

        return BSDFSample{
            .bsdf = eval(sgeom, λ, uv),
            .wi = ωi,
            .pdf = pdf(sgeom, λ),
            .did_refract = false,
        };
    }
}
