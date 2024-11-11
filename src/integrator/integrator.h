#ifndef PT_INTEGRATOR_H
#define PT_INTEGRATOR_H

#include "../embree_device.h"
#include "../math/vecmath.h"
#include "../render_context.h"
#include "../settings.h"
#include "../utils/basic_types.h"
#include "../utils/sampler.h"

class Integrator {
public:
    Integrator(const Settings &settings, RenderContext *rc, EmbreeDevice *device)
        : frame{settings.start_frame}, rc{rc}, settings{settings}, device{device} {}

    void
    integrate_pixel(uvec2 pixel) const {
        const auto dim = uvec2(rc->attribs.film.resx, rc->attribs.film.resy);

        const auto pixel_index = ((dim.y - 1U - pixel.y) * dim.x) + pixel.x;

        Sampler sampler{};
        sampler.init_frame(pixel, dim, frame, settings.spp);

        const auto cam_sample = sampler.sample2();
        const auto ray = gen_ray(pixel.x, pixel.y, dim.x, dim.y, cam_sample, rc->cam,
                                 rc->attribs.camera.camera_to_world);

        if (settings.render_normals) {
            const auto aov = render_aov(ray);
            rc->fb.get_pixels()[pixel_index] += aov;
            return;
        }

        SampledLambdas lambdas = SampledLambdas::new_sample_uniform(sampler.sample());

        const spectral radiance = integrator_mis_nee(ray, sampler, lambdas);

        if (radiance.is_invalid()) {
            spdlog::error("Invalid radiance {} at sample {}, pixel: {} x {} y",
                          radiance.to_str(), frame, pixel.x, pixel.y);
        }

        rc->fb.get_pixels()[pixel_index] +=
            lambdas.to_xyz(radiance) * rc->scene.attribs.film.iso / 100.F;
    }

    spectral
    integrator_mis_nee(Ray ray, Sampler &sampler, SampledLambdas &lambdas) const;

    spectral
    light_mis(const Scene &sc, const Intersection &its, const Ray &traced_ray,
              const LightSample &light_sample, const norm_vec3 &geom_normal,
              const Material *material, const spectral &throughput,
              const SampledLambdas &lambdas) const;

    vec3
    render_aov(const Ray &ray) const {
        const auto opt_its = device->cast_ray(ray);
        if (opt_its.has_value()) {
            return opt_its->normal;
        } else {
            return vec3(0.F);
        }
    }

    u32 frame = 0;

private:
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

    RenderContext *rc;
    Settings settings;
    EmbreeDevice *device;
};
#endif
