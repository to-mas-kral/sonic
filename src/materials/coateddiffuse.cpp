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

    return (1.F - fresnel_i) * sframe.nowi() / M_PIf;
}

bool
CoatedDifuseMaterial::is_dirac_delta() {
    return true;
}

spectral
CoatedDifuseMaterial::eval(const ShadingFrame &sframe, const SampledLambdas &lambdas,
                           const vec2 &uv) const {
    const f32 int_ior_s = int_ior.eval_single(lambdas[0]);
    const f32 ext_ior_s = ext_ior.eval_single(lambdas[0]);
    /// This is external / internal !
    const f32 rel_ior = ext_ior_s / int_ior_s;

    const f32 fresnel_i = fresnel_dielectric(1.F / rel_ior, sframe.nowi());

    if (sframe.noh() > 0.99999F) {
        // Specular case
        return spectral::make_constant(fresnel_i);
    } else {
        // Figure out the angle of the ray that refracts from inside to outisde wo
        const f32 cos_theta_out = sframe.nowo();
        const f32 sin_theta_out = std::sqrt(1.F - sqr(cos_theta_out));
        const f32 sin_theta_in = sin_theta_out * rel_ior;
        const f32 cos_theta_in = std::sqrt(1.F - sqr(sin_theta_in));
        const f32 fresnel_o = fresnel_dielectric(rel_ior, cos_theta_in);

        const auto a = fetch_reflectance(diffuse_reflectance, uv, lambdas);

        f32 re = 0.919317F;
        f32 ior_pow = int_ior_s;
        re -= (3.4793F / ior_pow);
        ior_pow *= ior_pow;
        re += (6.75335F / ior_pow);
        ior_pow *= ior_pow;
        re -= (7.80989F / ior_pow);
        ior_pow *= ior_pow;
        re += (4.98554F / ior_pow);
        ior_pow *= ior_pow;
        re -= (1.36881F / ior_pow);

        const f32 ri = 1.F - (sqr(rel_ior) * (1.F - re));

        const spectral scattering_brdf =
            a * (1.F - fresnel_i) * (1.F - fresnel_o) /
            ((spectral::make_constant(1.F) - a * ri) * M_PIf);

        const spectral brdf = scattering_brdf * sqr(rel_ior);

        return brdf;
    }
}

BSDFSample
CoatedDifuseMaterial::sample(const ShadingFrameIncomplete &sframe, const norm_vec3 &wo,
                             const vec3 &xi, SampledLambdas &lambdas,
                             const vec2 &uv) const {
    lambdas.terminate_secondary();
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

        return BSDFSample(spectral::make_constant(specular_brdf), potential_fresnel,
                          false, sframe_complete);
    } else {
        // sample diffuse
        const norm_vec3 wi = sample_cosine_hemisphere(vec2(xi.x, xi.y));
        const auto sframe_complete = ShadingFrame(sframe, wi, wo);

        return BSDFSample(eval(sframe_complete, lambdas, uv),
                          pdf(sframe_complete, lambdas), false, sframe_complete);
    }
}
