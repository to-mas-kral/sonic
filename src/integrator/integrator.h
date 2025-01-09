#ifndef PT_INTEGRATOR_H
#define PT_INTEGRATOR_H

#include "../embree_device.h"
#include "../math/vecmath.h"
#include "../render_context.h"
#include "../settings.h"
#include "../utils/basic_types.h"
#include "../utils/sampler.h"
#include "mis_nee_integrator.h"
#include "path_guiding_integrator.h"

// TODO: refactor together with RenderContext...

class Integrator {
public:
    static Integrator
    init(const Settings &settings, RenderContext *rc, EmbreeDevice *device) {
        switch (settings.integrator_type) {
        case IntegratorType::Naive:
            return Integrator(settings, rc, MisNeeIntegrator(settings, rc, device));
        case IntegratorType::MISNEE:
            return Integrator(settings, rc, MisNeeIntegrator(settings, rc, device));
        case IntegratorType::PathGuiding:
            return Integrator(settings, rc, PathGuidingIntegrator(settings, rc, device));
        default:
            panic("Erroneous integrator type.");
        }
    }

    // TODO: add back const after PG training refactor
    void
    integrate_pixel(uvec2 pixel) {
        const auto dim = uvec2(rc->attribs.film.resx, rc->attribs.film.resy);

        const auto pixel_index = ((dim.y - 1U - pixel.y) * dim.x) + pixel.x;

        Sampler sampler{};
        sampler.init_frame(pixel, dim, sample, settings.spp);

        const auto cam_sample = sampler.sample2();
        const auto ray = gen_ray(pixel.x, pixel.y, dim.x, dim.y, cam_sample, rc->cam,
                                 rc->attribs.camera.camera_to_world);

        SampledLambdas lambdas = SampledLambdas::new_sample_uniform(sampler.sample());

        spectral radiance;
        if (auto const *inner_integrator = std::get_if<MisNeeIntegrator>(&integrator)) {
            radiance = inner_integrator->radiance(ray, sampler, lambdas);
        } else if (auto *inner_integrator =
                       std::get_if<PathGuidingIntegrator>(&integrator)) {
            radiance = inner_integrator->radiance_training(ray, sampler, lambdas);
        } else {
            panic("Wrong integrator type");
        }

        if (radiance.is_invalid()) {
            spdlog::error("Invalid radiance {} at sample {}, pixel: {} x {} y",
                          radiance.to_str(), sample, pixel.x, pixel.y);
        }

        rc->fb.get_pixels()[pixel_index] +=
            lambdas.to_xyz(radiance) * rc->scene.attribs.film.iso / 100.F;
    }

    u32 sample = 0;

private:
    Integrator(const Settings &settings, RenderContext *rc,
               const MisNeeIntegrator &integrator)
        : sample{settings.start_frame}, integrator{integrator}, rc{rc},
          settings{settings} {}

    Integrator(const Settings &settings, RenderContext *rc,
               const PathGuidingIntegrator &integrator)
        : sample{settings.start_frame}, integrator{integrator}, rc{rc},
          settings{settings} {}

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

    std::variant<MisNeeIntegrator, PathGuidingIntegrator> integrator;

    RenderContext *rc;
    Settings settings;
};
#endif
