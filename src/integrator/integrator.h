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
    Integrator(Settings settings, RenderContext *rc, EmbreeDevice *device)
        : rc{rc}, settings{settings}, device{device} {}

    void
    integrate_pixel(uvec2 pixel) const {
        uvec2 dim =
            uvec2(rc->attribs.camera_attribs.resx, rc->attribs.camera_attribs.resy);

        auto pixel_index = ((dim.y - 1U - pixel.y) * dim.x) + pixel.x;

        Sampler sampler{};
        sampler.init_frame(uvec2(pixel.x, pixel.y), uvec2(dim.x, dim.y), frame);

        auto cam_sample = sampler.sample2();
        auto ray = gen_ray(pixel.x, pixel.y, dim.x, dim.y, cam_sample, rc->cam,
                           rc->attribs.camera_attribs.camera_to_world);

        SampledLambdas lambdas = SampledLambdas::new_sample_uniform(sampler.sample());

        spectral radiance = integrator_mis_nee(ray, sampler, lambdas);

        if (std::isnan(radiance[0]) || std::isinf(radiance[0])) {
            fmt::println("\nerr at frame {} x {} y {}", frame, pixel.x, pixel.y);
        }

        rc->fb.get_pixels()[pixel_index] += lambdas.to_xyz(radiance);
    }

    spectral
    integrator_mis_nee(Ray ray, Sampler &sampler, const SampledLambdas &lambdas) const;

    spectral
    light_mis(const Scene &sc, const Intersection &its, const Ray &traced_ray,
              const LightSample &light_sample, const norm_vec3 &geom_normal,
              const ShapeSample &shape_sample, const Material *material,
              const spectral &throughput, const SampledLambdas &lambdas) const;

    u32 frame = 0;

private:
    static Ray
    gen_ray(u32 x, u32 y, u32 res_x, u32 res_y, const vec2 &sample, const Camera &cam,
            const mat4 &cam_to_world) {
        f32 image_x = static_cast<f32>(res_x - 1U);
        f32 image_y = static_cast<f32>(res_y - 1U);

        f32 s_offset = sample.x;
        f32 t_offset = sample.y;

        f32 s = (static_cast<f32>(x) + s_offset) / image_x;
        f32 t = (static_cast<f32>(y) + t_offset) / image_y;

        Ray ray = cam.get_ray(s, t);
        ray.transform(cam_to_world);

        return ray;
    }

    RenderContext *rc;
    Settings settings;
    EmbreeDevice *device;
};
#endif
