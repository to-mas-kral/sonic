#ifndef PT_BDPT_NEE_INTEGRATOR_CPP
#define PT_BDPT_NEE_INTEGRATOR_CPP

#include "integrator.h"

spectral
Integrator::mis_xp_y1_y0(const Intersection &xp_its, const Intersection &y1_its,
                         const SampledLambdas &lambdas,
                         const std::vector<Texture> &textures, const vec2 &uv,
                         const point3 &y0, const norm_vec3 &xp_wo) {
    norm_vec3 xp_y1_dir = (y1_its.pos - xp_its.pos).normalized();

    bool xp_y1_visible = vec3::dot(y1_its.normal, -xp_y1_dir) > 0.f &&
                         vec3::dot(xp_its.normal, xp_y1_dir) > 0.f;

    if (xp_y1_visible) {
        point3 ray_orig = offset_ray(y1_its.pos, y1_its.geometric_normal);
        xp_y1_visible = device->is_visible(ray_orig, xp_its.pos);
    }

    if (!xp_y1_visible) {
        return spectral::ZERO();
    }

    auto &materials = rc->scene.materials;
    auto xp_material = &materials[xp_its.material_id];
    auto y1_material = &materials[y1_its.material_id];

    auto xp_sgeom = ShadingGeometry::make(xp_its.normal, xp_y1_dir, xp_wo);
    auto y1_sgeom =
        ShadingGeometry::make(y1_its.normal, (y0 - y1_its.pos).normalized(), -xp_y1_dir);

    // TODO: fix UVs
    spectral xp_brdf = xp_material->eval(xp_sgeom, lambdas, textures.data(), uv);
    spectral y1_brdf = y1_material->eval(y1_sgeom, lambdas, textures.data(), uv);

    spectral res = xp_brdf * xp_sgeom.cos_theta * y1_brdf * y1_sgeom.cos_theta;

    if (res[0] < 0.f) {
        fmt::println("{} {} {} {}", xp_brdf[0], xp_sgeom.cos_theta, y1_brdf[1],
                     y1_sgeom.cos_theta);
    }
    assert(res[0] >= 0.f);
    assert(!std::isnan(res[0]));
    assert(!std::isinf(res[0]));

    return res;
}

spectral
Integrator::integrator_bdpt_nee(Ray ray, Sampler &sampler,
                                const SampledLambdas &lambdas) {
    auto &sc = rc->scene;
    auto &materials = sc.materials;
    auto &lights = sc.lights;
    auto &textures = sc.textures;
    auto max_depth = rc->attribs.max_depth;

    spectral radiance = spectral::ZERO();
    u32 depth = 1;
    f32 last_pdf_bxdf = 0.f;

    bool xp_is_dirac_delta = false;
    norm_vec3 xp_wo{0.f, 1.f, 0.f};
    spectral xp_throughput = spectral::ONE();
    Intersection xp_its = Intersection::make_empty();

    bool xi_is_dirac_delta = false;
    norm_vec3 xi_wo = -ray.dir;
    spectral xi_throughput = spectral::ONE();
    Intersection xi_its = Intersection::make_empty();

    while (true) {
        auto opt_its = device->cast_ray(ray);
        if (!opt_its.has_value()) {
            if (!sc.has_envmap) {
                break;
            } else {
                const Envmap *envmap = &sc.envmap;
                spectral envrad = envmap->get_ray_radiance(ray, lambdas);

                // TODO: do envmap sampling...
                radiance += envrad;
                break;
            }
        }

        auto its = opt_its.value();
        auto bsdf_sample_rand = sampler.sample3();
        auto rr_sample = sampler.sample();

        auto material = &materials[its.material_id];
        bool is_frontfacing = vec3::dot(-ray.dir, its.normal) >= 0.f;

        if (!is_frontfacing && !material->is_twosided) {
            break;
        }

        if (!is_frontfacing) {
            its.normal = -its.normal;
            its.geometric_normal = -its.geometric_normal;
        }

        if (its.has_light && is_frontfacing) {
            spectral emission = lights[its.light_id].emitter.emission(lambdas);

            if (depth < 2 || xi_is_dirac_delta || xp_is_dirac_delta) {
                radiance += xi_throughput * emission;
                // Primary ray hit, can't apply MIS...
                // radiance += throughput * emission;
            } else {
                /*auto bxdf_mis_contrib = bxdf_mis(sc, throughput, last_hit_pos,
                                                 last_pdf_bxdf, its, emission);

                radiance += bxdf_mis_contrib;*/

                // TODO:
                // radiance += mis_xp_xc_xn();
            }
        }

        // Do this before light sampling, because that "extends the path"
        if (max_depth > 0 && depth >= max_depth) {
            break;
        }

        // TODO: FIMXE: update order
        // Update
        xp_its = xi_its;
        xi_its = its;

        xp_is_dirac_delta = xi_is_dirac_delta;
        xi_is_dirac_delta = material->is_dirac_delta();
        if (!xp_is_dirac_delta && !xi_is_dirac_delta && depth >= 2) {
            f32 light_sample = sampler.sample();
            auto sampled_light = sc.sample_lights(light_sample);
            if (sampled_light.has_value()) {
                auto shape_rng = sampler.sample3();
                auto shape_sample = sc.geometry.sample_shape(
                    sampled_light.value().light.shape, its.pos, shape_rng);

                // TODO:
                // radiance += mis_xp_xc_y0();

                /*
                 * Sample y1
                 * */

                // TODO: fix independent sampling order
                // TODO: assuming light has a diffuse BRDF... common
                norm_vec3 sample_dir = sample_cosine_hemisphere(sampler.sample2());
                norm_vec3 wi = orient_dir(sample_dir, shape_sample.normal);

                auto y1_ray = spawn_ray(shape_sample.pos, shape_sample.normal, wi);

                auto opt_its_y1 = device->cast_ray(y1_ray);
                if (opt_its_y1.has_value()) {
                    Intersection y1_its = opt_its_y1.value();

                    bool is_y1_frontfacing = vec3::dot(-wi, y1_its.normal) >= 0.f;
                    if (!is_y1_frontfacing) {
                        y1_its.normal = -y1_its.normal;
                        y1_its.geometric_normal = -y1_its.geometric_normal;
                    }

                    auto y1_material = &materials[y1_its.material_id];

                    if (is_y1_frontfacing || y1_material->is_twosided) {
                        spectral emission =
                            sampled_light.value().light.emitter.emission(lambdas);

                        norm_vec3 xp_y0 = (shape_sample.pos - xp_its.pos).normalized();

                        f32 xp_y0_cos_theta = vec3::dot(shape_sample.normal, -xp_y0);

                        if (xp_y0_cos_theta > 0.f) {
                            f32 xp_y0_magsq =
                                (xp_its.pos - shape_sample.pos).length_squared();
                            // TODO!!!: currently calculating the shape PDF by
                            // assuming pdf = 1. / area
                            //  will have to change with non-uniform sampling !
                            f32 pdf_y0 = shape_sample.pdf * sampled_light.value().pdf *
                                         xp_y0_magsq / xp_y0_cos_theta;

                            f32 pdf_y1 = vec3::dot(wi, shape_sample.normal) / M_PIf;

                            spectral xp_y1_y0_throu =
                                mis_xp_y1_y0(xp_its, y1_its, lambdas, textures, xi_its.uv,
                                             shape_sample.pos, xp_wo);

                            if (pdf_y0 <= 0.f) {
                                fmt::println("{} {} {}", pdf_y0, xp_y0_magsq,
                                             vec3::dot(shape_sample.normal, -xp_y0));
                            }

                            assert(pdf_y0 > 0.f);
                            assert(pdf_y1 > 0.f);
                            assert(xp_throughput[0] >= 0.f);

                            spectral bidir_contrib = xp_y1_y0_throu * emission *
                                                     xp_throughput / (pdf_y0 * pdf_y1);

                            assert(bidir_contrib[0] >= 0.f);
                            assert(!std::isnan(bidir_contrib[0]));
                            assert(!std::isinf(bidir_contrib[0]));

                            radiance += bidir_contrib;
                        }
                    }
                }

                // TODO: light sample, create y1
                /*auto light_mis_contrib = light_mis(
                    params, its, ray, sampled_light.value(),
                its.geometric_normal, shape_sample, material, throughput,
                lambdas);

                radiance += light_mis_contrib;*/
            }
        }

        auto bsdf_sample_opt =
            material->sample(its.normal, -ray.dir, bsdf_sample_rand, lambdas,
                             textures.data(), its.uv, is_frontfacing);

        if (!bsdf_sample_opt.has_value()) {
            break;
        }
        auto bsdf_sample = bsdf_sample_opt.value();

        auto sgeom_bxdf = ShadingGeometry::make(its.normal, bsdf_sample.wi, -ray.dir);

        auto spawn_ray_normal =
            (bsdf_sample.did_refract) ? -its.geometric_normal : its.geometric_normal;
        Ray bxdf_ray = spawn_ray(its.pos, spawn_ray_normal, bsdf_sample.wi);

        auto rr = russian_roulette(depth, rr_sample, xi_throughput);
        if (!rr.has_value()) {
            break;
        }

        auto roulette_compensation = rr.value();

        xp_throughput = xi_throughput;
        xi_throughput *= bsdf_sample.bsdf * sgeom_bxdf.cos_theta *
                         (1.f / (bsdf_sample.pdf * roulette_compensation));

        ray = bxdf_ray;

        xp_wo = xi_wo;
        xi_wo = -bxdf_ray.dir;

        last_pdf_bxdf = bsdf_sample.pdf;
        depth++;

        if (depth == 1024) {
            fmt::println("Specular infinite self-intersection path\n");
            // FIXME: specular infinite path caused by self-intersections
            break;
        }
    }

    return radiance;
}

#endif // PT_BDPT_NEE_INTEGRATOR_CPP
