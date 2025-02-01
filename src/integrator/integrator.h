#ifndef PT_INTEGRATOR_H
#define PT_INTEGRATOR_H

#include "../embree_accel.h"
#include "../integrator_context.h"
#include "../math/vecmath.h"
#include "../settings.h"
#include "../utils/basic_types.h"
#include "../utils/sampler.h"

struct PathVertex {
    spectral throughput{spectral::ONE()};
    f32 pdf_bxdf{0.F};
    bool is_bxdf_delta{false};
    point3 pos{0.F};
};

struct IterationProgressInfo {
    u32 samples_max;
    u32 samples_done;
};

class Integrator {
public:
    Integrator(IntegratorContext *const ctx, const Settings &settings)
        : ctx(ctx), settings(settings) {}

    void
    integrate_pixel(uvec2 pixel) {
        const auto dim = uvec2(ctx->attribs().film.resx, ctx->attribs().film.resy);

        Sampler sampler{};
        sampler.init_frame(pixel, dim, sample, settings.spp);

        const auto cam_sample = sampler.sample2();
        const auto ray = gen_ray(pixel.x, pixel.y, dim.x, dim.y, cam_sample, ctx->cam(),
                                 ctx->attribs().camera.camera_to_world);

        SampledLambdas lambdas = SampledLambdas::new_sample_importance(sampler);

        const auto radiance = estimate_radiance(ray, sampler, lambdas, pixel);

        if (radiance.is_invalid()) {
            spdlog::error("Invalid radiance {} at sample {}, pixel: {} x {} y",
                          radiance.to_str(), sample, pixel.x, pixel.y);
        }

        ctx->framebuf().add_to_pixel(pixel, lambdas.to_xyz(radiance) *
                                                ctx->attribs().film.iso / 100.F);
    }

    virtual void
    next_sample() {
        sample++;
        ctx->framebuf().num_samples++;
    }

    virtual std::optional<IterationProgressInfo>
    iter_progress_info() {
        return {};
    }

    Integrator(const Integrator &other) = default;

    Integrator(Integrator &&other) noexcept = default;

    Integrator &
    operator=(const Integrator &other) = default;

    Integrator &
    operator=(Integrator &&other) noexcept = default;

    virtual ~Integrator() = default;

protected:
    spectral
    light_mis(const Intersection &its, const Ray &traced_ray,
              const LightSample &light_sample, const norm_vec3 &geom_normal,
              const Material *material, const spectral &throughput,
              const SampledLambdas &lambdas) const;

    spectral
    bxdf_mis(const spectral &throughput, const point3 &last_hit_pos, f32 last_pdf_bxdf,
             const Intersection &its, const spectral &emission) const;

    void
    add_radiance_contrib(spectral &radiance, const spectral &contrib) const;

    void
    record_aovs(const uvec2 &pixel, const Intersection &its) const {
        ctx->framebuf().add_aov(pixel, "Normals", its.normal.as_tuple());
        ctx->framebuf().add_aov(pixel, "Position", its.pos.as_tuple());
    }

    IntegratorContext *ctx;
    Settings settings;

    u32 sample = 0;

private:
    virtual spectral
    estimate_radiance(Ray ray, Sampler &sampler, SampledLambdas &lambdas,
                      uvec2 &pixel) = 0;

    static Ray
    gen_ray(const u32 x, const u32 y, const u32 res_x, const u32 res_y,
            const vec2 &sample, const Camera &cam, const mat4 &cam_to_world) {
        const f32 image_x = static_cast<f32>(res_x - 1U);
        const f32 image_y = static_cast<f32>(res_y - 1U);

        const f32 s_offset = sample.x;
        const f32 t_offset = sample.y;

        const f32 s = (static_cast<f32>(x) + s_offset) / image_x;
        const f32 t = (static_cast<f32>(y) + t_offset) / image_y;

        Ray ray = cam.get_ray(s, t);
        ray.transform(cam_to_world);

        return ray;
    }
};
#endif
