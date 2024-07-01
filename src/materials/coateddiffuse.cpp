#include "coateddiffuse.h"

#include "../integrator/shading_frame.h"
#include "../math/sampling.h"
#include "common.h"

f32
CoatedDifuseMaterial::pdf(const ShadingFrame &sframe,
                          const SampledLambdas &lambdas) const {
    const f32 int_η = int_ior.eval_single(lambdas[0]);
    const f32 ext_η = ext_ior.eval_single(lambdas[0]);
    const f32 rel_η = int_η / ext_η;

    const f32 fresnel_i = fresnel_dielectric(rel_η, sframe.nowo());

    return (1.f - fresnel_i) * sframe.nowi() / M_PIf;
}

bool
CoatedDifuseMaterial::is_dirac_delta() {
    return true;
}

spectral
CoatedDifuseMaterial::eval(const ShadingFrame &sframe,
                           const SampledLambdas &lambdas, const vec2 &uv) const {
    // TODO: fix this and roughcoateddiffuse for multiple spectral samples.. this should
    //       evaluate separately for all lambdas
    const f32 int_ior_s = int_ior.eval_single(lambdas[0]);
    const f32 ext_ior_s = ext_ior.eval_single(lambdas[0]);
    /// This is external / internal !
    const f32 rel_ior = ext_ior_s / int_ior_s;

    const f32 fresnel_i = fresnel_dielectric(1.f / rel_ior, sframe.nowi());

    if (sframe.noh() > 0.99999f) {
        // Specular case
        return spectral::make_constant(fresnel_i);
    } else {
        // Figure out the angle of the ray that refracts from inside to outisde wo
        const f32 cos_theta_out = sframe.nowo();
        const f32 sin_theta_out = std::sqrt(1.f - sqr(cos_theta_out));
        const f32 sin_theta_in = sin_theta_out * rel_ior;
        const f32 cos_theta_in = std::sqrt(1.f - sqr(sin_theta_in));
        const f32 fresnel_o = fresnel_dielectric(rel_ior, cos_theta_in);

        const auto a = fetch_reflectance(diffuse_reflectance, uv, lambdas);

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
            a * (1.f - fresnel_i) * (1.f - fresnel_o) /
            ((spectral::make_constant(1.f) - a * ri) * M_PIf);

        const spectral brdf = scattering_brdf * sqr(rel_ior);

        return brdf;
    }
}

BSDFSample
CoatedDifuseMaterial::sample(const ShadingFrameIncomplete &sframe, const norm_vec3 &wo,
                             const vec3 &xi, const SampledLambdas &lambdas,
                             const vec2 &uv) const {
    const f32 int_η = int_ior.eval_single(lambdas[0]);
    const f32 ext_η = ext_ior.eval_single(lambdas[0]);
    const f32 rel_η = int_η / ext_η;

    const f32 nowo = ShadingFrameIncomplete::cos_theta(wo);
    const f32 potential_fresnel = fresnel_dielectric(rel_η, nowo);

    if (xi.z < potential_fresnel) {
        // sample specular
        const norm_vec3 wi = ShadingFrameIncomplete::reflect(wo).normalized();
        const auto sframe_complete = ShadingFrame(sframe, wi, wo);

        const f32 specular_brdf = potential_fresnel / sframe_complete.nowi();

        return BSDFSample{
            .bsdf = spectral::make_constant(specular_brdf),
            .pdf = potential_fresnel,
            .did_refract = false,
            .sframe = sframe_complete,
        };
    } else {
        // sample diffuse
        const norm_vec3 wi = sample_cosine_hemisphere(vec2(xi.x, xi.y));
        const auto sframe_complete = ShadingFrame(sframe, wi, wo);

        return BSDFSample{
            .bsdf = eval(sframe_complete, lambdas, uv),
            .pdf = pdf(sframe_complete, lambdas),
            .did_refract = false,
            .sframe = sframe_complete,
        };
    }
}
