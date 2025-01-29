
#include "integrator.h"

// Multiple Importance Sampling for lights
spectral
Integrator::light_mis(const Intersection &its, const Ray &traced_ray,
                      const LightSample &light_sample, const norm_vec3 &geom_normal,
                      const Material *material, const spectral &throughput,
                      const SampledLambdas &lambdas) const {
    const point3 light_pos = light_sample.pos;
    const norm_vec3 pl = (light_pos - its.pos).normalized();
    // TODO: move this into light sampling itself
    const f32 cos_light = vec3::dot(light_sample.normal, -pl);

    const auto sframe_light = ShadingFrame(its.normal, pl, -traced_ray.dir());
    if (sframe_light.is_degenerate()) {
        return spectral::ZERO();
    }

    // Quickly precheck if light is reachable
    if (sframe_light.nowi() > 0.F && cos_light > 0.F) {
        const point3 ray_orig = offset_ray(its.pos, geom_normal);
        const spectral bxdf_light = material->eval(sframe_light, lambdas, its.uv);
        // eval bxdf and check if it's 0 to potentially avoid the raycast
        if (bxdf_light.is_zero()) {
            return spectral::ZERO();
        }

        if (ctx->accel().is_visible(ray_orig, light_pos)) {
            const f32 mat_pdf = material->pdf(sframe_light, lambdas, its.uv);

            const f32 weight_light = mis_power_heuristic(light_sample.pdf, mat_pdf);

            const auto contrib = bxdf_light * sframe_light.abs_nowi() *
                                 (1.F / light_sample.pdf) * light_sample.emission *
                                 weight_light * throughput;

            assert(!contrib.is_invalid());
            return contrib;
        }
    }

    return spectral::ZERO();
}

spectral
Integrator::bxdf_mis(const spectral &throughput, const point3 &last_hit_pos,
                     const f32 last_pdf_bxdf, const Intersection &its,
                     const spectral &emission) const {
    const norm_vec3 pl_norm = (its.pos - last_hit_pos).normalized();
    const f32 pl_mag_sq = (its.pos - last_hit_pos).length_squared();
    const f32 cos_light = vec3::dot(its.normal, -pl_norm);

    // last_pdf_bxdf is the probability of this light having been sampled
    // from the probability distribution of the BXDF of the *preceding*
    // hit.

    // TODO: !!!currently calculating the light PDF by assuming pdf = 1. / area
    //  will have to change with non-uniform sampling !
    const f32 shape_pdf =
        1.F / ctx->scene().lights[its.light_id].area(ctx->scene().geometry_container);

    // pdf_light is the probability of this point being sampled from the
    // probability distribution of the lights.
    const f32 pdf_light = ctx->scene().light_sampler.light_sample_pdf(its.light_id) *
                          pl_mag_sq * shape_pdf / (cos_light);

    const f32 bxdf_weight = mis_power_heuristic(last_pdf_bxdf, pdf_light);
    // pdf is already contained in the throughput
    const auto contrib = throughput * emission * bxdf_weight;
    return contrib;
}

void
Integrator::add_radiance_contrib(spectral &radiance, const spectral &contrib) const {
    if (contrib.max_component() > 10000.F) {
        spdlog::warn("potential firefly");
    }
    assert(!contrib.is_invalid());
    assert(!radiance.is_invalid());
    radiance += contrib;
}
