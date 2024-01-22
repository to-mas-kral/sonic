#ifndef PT_MATERIAL_H
#define PT_MATERIAL_H

#include "../color/sampled_spectrum.h"
#include "../color/spectrum.h"
#include "../integrator/utils.h"
#include "../math/sampling.h"
#include "../math/vecmath.h"
#include "../texture.h"
#include "../utils/basic_types.h"

#include <cuda/std/complex>

enum class MaterialType : u8 {
    Diffuse,
    Conductor,
    RoughConductor,
    Dielectric,
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
    make_conductor(Spectrum eta, Spectrum k) {
        return Material{.type = MaterialType::Conductor,
                        .conductor = {
                            .eta = eta,
                            .k = k,
                        }};
    }

    static bool
    is_alpha_effectively_zero(f32 alpha) {
        return alpha < 0.001f;
    }

    static Material
    make_rough_conductor(f32 alpha, Spectrum eta, Spectrum k) {
        if (is_alpha_effectively_zero(alpha)) {
            return Material{.type = MaterialType::Conductor,
                            .conductor = {
                                .eta = eta,
                                .k = k,
                            }};
        } else {
            return Material{.type = MaterialType::RoughConductor,
                            .rough_conductor = {
                                .eta = eta,
                                .k = k,
                                .alpha = alpha,
                            }};
        }
    }

    __device__ BSDFSample
    sample_diffuse(const norm_vec3 &normal, const norm_vec3 &wo, const vec2 &sample,
                   const SampledLambdas &lambdas, const Texture *textures,
                   const vec2 &uv) const {
        norm_vec3 sample_dir = sample_cosine_hemisphere(sample);
        norm_vec3 wi = orient_dir(sample_dir, normal);
        auto sgeom = ShadingGeometry::make(normal, wi, wo);
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
        auto sgeom = ShadingGeometry::make(normal, wi, wo);
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
            auto sgeom = ShadingGeometry::make(normal, wi, wo);

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
                auto sgeom = ShadingGeometry::make(normal, wi, wo);
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

    static __host__ __device__ f32
    distribution_ggx(f32 noh, f32 alpha) {
        if (noh < 0.f) {
            return 0.f;
        }

        f32 asq = sqr(alpha);
        f32 denom = sqr(noh) * (asq - 1.f) + 1.f;

        return asq / (M_PIf * sqr(denom));
    }

    static __host__ __device__ f32
    geometry_ggx(f32 now, f32 how, f32 alpha) {
        if (how / now <= 0.f) {
            return 0.f;
        }

        f32 asq = sqr(alpha);
        f32 denom = now + sqrt(asq + (1.f - asq) * sqr(now));
        return (2.f * now) / denom;
    }

    static __host__ __device__ f32
    vndf_ggx(const ShadingGeometry &sgeom, f32 alpha) {
        if (sgeom.noh < 0.f) {
            return 0.f;
        }

        f32 g1 = geometry_ggx(sgeom.nowo, sgeom.howo, alpha);
        f32 d = distribution_ggx(sgeom.noh, alpha);
        assert(g1 >= 0.f && g1 <= 1.f);
        assert(d >= 0.f);

        return (g1 / abs(sgeom.nowo)) * d * max(sgeom.howo, 0.f);
    }

    static __device__ f32
    visibility_smith_height_correlated_ggx(f32 nowo, f32 nowi, f32 alpha) {
        f32 asq = alpha * alpha;
        f32 NoVsq = nowo * nowo;
        f32 NoLsq = nowi * nowi;

        f32 denoml = nowi * sqrt(asq + NoVsq * (1.f - asq));
        f32 denomv = nowo * sqrt(asq + NoLsq * (1.f - asq));

        // TODO: better Protect against division by zero
        if (denoml + denomv == 0.f) {
            printf("nowi: %f nowo: %f\n\n", nowi, nowo);
        }
        return 0.5f / (denoml + denomv);
    }

    /// Taken from PBRTv4
    __host__ __device__ static f32
    fresnel_conductor(cuda::std::complex<f32> rel_ior, f32 cos_theta_i) {
        using complex = cuda::std::complex<f32>;
        f32 sin2_theta_i = 1.f - sqr(cos_theta_i);
        complex sin2_theta_t = sin2_theta_i / sqr(rel_ior);

        complex cos_theta_t = sqrt(1.f - sin2_theta_t);
        complex r_parl =
            (rel_ior * cos_theta_i - cos_theta_t) / (rel_ior * cos_theta_i + cos_theta_t);
        complex r_perp =
            (cos_theta_i - rel_ior * cos_theta_t) / (cos_theta_i + rel_ior * cos_theta_t);
        return (norm(r_parl) + norm(r_perp)) / 2.f;
    }

    __host__ __device__ spectral
    eval_torrance_sparrow(const ShadingGeometry &sgeom,
                          const SampledLambdas &lambdas) const {
        f32 alpha = rough_conductor.alpha;
        // TODO: have to store the current IOR... when it isn't 1...
        spectral rel_ior = rough_conductor.eta.eval(lambdas);
        spectral k = rough_conductor.k.eval(lambdas);

        f32 D = distribution_ggx(sgeom.noh, alpha);

        spectral fresnel = spectral::ZERO();
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            fresnel[i] =
                fresnel_conductor(cuda::std::complex<f32>(rel_ior[i], k[i]), sgeom.howo);
        }

        float G = geometry_ggx(sgeom.nowi, sgeom.howo, alpha) *
                  geometry_ggx(sgeom.nowo, sgeom.howo, alpha);
        return (fresnel * G * D) / (4.f * sgeom.nowo * sgeom.nowi);

        // TODO: try the height-correlated smith
        /*f32 V = visibility_smith_height_correlated_ggx(sgeom.nowo, sgeom.nowi, alpha);
        return fresnel * V * D;*/
    }

    // Taken from "Sampling Visible GGX Normals with Spherical Caps - Jonathan Dupuy"
    static __host__ __device__ vec3
    sample_vndf_hemisphere(vec2 u, vec3 wi) {
        // sample a spherical cap in (-wi.z, 1]
        float phi = 2.f * M_PIf * u.x;
        float z = fma((1.f - u.y), (1.f + wi.z), -wi.z);
        float sin_theta = sqrt(cuda::std::clamp(1.f - z * z, 0.f, 1.f));
        float x = sin_theta * cos(phi);
        float y = sin_theta * sin(phi);
        vec3 c = vec3(x, y, z);
        // compute halfway direction;
        vec3 h = c + wi;
        // return without normalization as this is done later (see line 25)
        return h;
    }

    // Adapted from "Sampling Visible GGX Normals with Spherical Caps - Jonathan Dupuy"
    __host__ __device__ COption<BSDFSample>
    sample_trowbridge_reitz(const norm_vec3 &normal, const norm_vec3 &wo,
                            const vec2 &sample, const SampledLambdas &lambdas,
                            const Texture *textures, const vec2 &uv) const {
        // TODO: handle case when alpha ~= 0 (brdf degenerates to dirac delta)
        f32 alpha = rough_conductor.alpha;

        auto [b0_, b1_, b2_] = coordinate_system(normal);
        auto bz = b0_.normalized();
        auto bx = b1_.normalized();
        auto by = b2_.normalized();

        norm_vec3 wo_sp =
            vec3(vec3::dot(wo, bx), vec3::dot(wo, by), vec3::dot(wo, bz)).normalized();

        // warp to the hemisphere configuration
        vec3 wi_std = vec3(wo_sp.x * alpha, wo_sp.y * alpha, wo_sp.z).normalized();
        // sample the hemisphere
        vec3 wm_std = sample_vndf_hemisphere(sample, wi_std);
        // warp back to the ellipsoid configuration
        norm_vec3 wm = vec3(wm_std.x * alpha, wm_std.y * alpha, wm_std.z).normalized();

        norm_vec3 wm_rs = vec3(wm.x * bx + wm.y * by + wm.z * bz).normalized();
        norm_vec3 wi = vec3::reflect(wo, wm_rs).normalized();

        auto sgeom = ShadingGeometry::make(normal, wi, wo);

        if (sgeom.nowi * sgeom.nowo <= 0.f) {
            return {};
        }

        auto s = BSDFSample{
            .bsdf = eval_torrance_sparrow(sgeom, lambdas),
            .wi = wi,
            .pdf = pdf(sgeom),
            .did_refract = false,
        };

        return s;
    }

    __device__ COption<BSDFSample>
    sample(const norm_vec3 &normal, const norm_vec3 &wo, const vec2 &sample,
           const SampledLambdas &lambdas, const Texture *textures, const vec2 &uv,
           bool is_frontfacing) const {
        switch (type) {
        case MaterialType::Diffuse:
            return sample_diffuse(normal, wo, sample, lambdas, textures, uv);
        case MaterialType::Conductor:
            return sample_conductor(normal, wo, lambdas, textures, uv);
        case MaterialType::RoughConductor:
            return sample_trowbridge_reitz(normal, wo, sample, lambdas, textures, uv);
        case MaterialType::Dielectric:
            return sample_dielectric(normal, wo, sample, lambdas, textures, uv,
                                     is_frontfacing);
        }
    }

    __host__ __device__ f32
    pdf_diffuse(const ShadingGeometry &sgeom) const {
        return sgeom.cos_theta / M_PIf;
    }

    __host__ __device__ f32
    pdf_conductor() const {
        return 0.f;
    }

    __host__ __device__ f32
    pdf_roughconductor(const ShadingGeometry &sgeom) const {
        f32 alpha = rough_conductor.alpha;

        if (sgeom.noh < 0.f) {
            return 0.f;
        }

        f32 g1 = geometry_ggx(sgeom.nowo, sgeom.howo, alpha);
        f32 d = distribution_ggx(sgeom.noh, alpha);
        return g1 * d / (4.f * abs(sgeom.nowo));
    }

    __host__ __device__ f32
    pdf_dielectric() const {
        return 0.f;
    }

    // Probability density function of sampling the BRDF
    __host__ __device__ f32
    pdf(const ShadingGeometry &sgeom) const {
        switch (type) {
        case MaterialType::Diffuse:
            return pdf_diffuse(sgeom);
        case MaterialType::Conductor:
            return pdf_conductor();
        case MaterialType::RoughConductor:
            return pdf_roughconductor(sgeom);
        case MaterialType::Dielectric:
            return pdf_dielectric();
        }
    }

    __device__ spectral
    eval_diffuse(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
                 const Texture *textures, const vec2 &uv) const {
        spectral refl{};

        // TODO: encapsulate textures
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

    __device__ spectral
    eval_conductor(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
                   const Texture *textures, const vec2 &uv) const {
        // TODO: have to store the current IOR... when it isn't 1...
        spectral rel_ior = rough_conductor.eta.eval(lambdas);
        spectral k = rough_conductor.k.eval(lambdas);

        spectral fresnel = spectral::ZERO();
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            fresnel[i] =
                fresnel_conductor(cuda::std::complex<f32>(rel_ior[i], k[i]), sgeom.howo);
        }
        return fresnel / sgeom.cos_theta;
    }

    __device__ spectral
    eval_dielectric() const {
        // This should only be evaluated during sampling
        assert(false);
        return spectral::make_constant(NAN);
    }

    __device__ spectral
    eval(const ShadingGeometry &sgeom, const SampledLambdas &lambdas,
         const Texture *textures, const vec2 &uv) const {
        switch (type) {
        case MaterialType::Diffuse:
            return eval_diffuse(sgeom, lambdas, textures, uv);
        case MaterialType::RoughConductor:
            return eval_torrance_sparrow(sgeom, lambdas);
        case MaterialType::Conductor:
            return eval_conductor(sgeom, lambdas, textures, uv);
        case MaterialType::Dielectric:
            return eval_dielectric();
        }
    }

    __device__ bool
    is_dirac_delta() const {
        switch (type) {
        case MaterialType::Diffuse:
            return false;
        case MaterialType::Conductor:
            return true;
        case MaterialType::RoughConductor:
            return false;
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
        struct {
            // real part of the IOR
            Spectrum eta;
            // absorption coefficient
            Spectrum k;
            f32 alpha;
        } rough_conductor;
        struct {
            // real part of the IOR
            Spectrum eta;
            // absorption coefficient
            Spectrum k;
        } conductor;
    };
};

#endif // PT_MATERIAL_H
