
#include "megakernel.h"

#include "../geometry/intersection.h"
#include "../integrator/utils.h"
#include "../utils/rng.h"
#include "raygen.h"

// (s, t) are coords in screen space
__device__ vec3 render(RenderContext *rc, u32 x, u32 y) {
    auto pixel_index = rc->fb.pixel_index(x, y);
    auto sampler = &rc->fb.get_rand_state()[pixel_index];

    u32 resx = rc->fb.get_res_x();
    u32 resy = rc->fb.get_res_y();

    auto cam_sample = vec2(sampler->sample(), sampler->sample());
    auto bsdf_sample = vec2(sampler->sample(), sampler->sample());
    auto rr_sample = sampler->sample();
    Ray ray = gen_ray(x, y, resx, resy, cam_sample, rc);

    /*
     * Iterative naive path tracing
     * */
    u32 depth = 1;
    vec3 throughput = vec3(1.f);
    vec3 radiance = vec3(0.f);

    while (true) {
        // rc->ray_counter.fetch_add(1);
        auto pot_its = rc->intersect_scene(ray);
        if (pot_its.has_value()) {
            auto its = pot_its.value();
            auto material = &rc->materials[its.mesh->material_id];

            vec3 emission = vec3(0.f);
            if (its.mesh->has_light) {
                auto light_id = its.mesh->light_id;
                emission = rc->lights[light_id].emission();
            }

            if (glm::dot(-ray.dir, its.normal) < 0.f) {
                its.normal = -its.normal;
                emission = vec3(0.f);
            }

            vec3 sample_dir = material->sample(its.normal, -ray.dir, bsdf_sample);
            // TODO: what to do when cos_theta is 0 ? this minimum value is a band-aid
            // solution...
            f32 cos_theta = max(glm::dot(its.normal, sample_dir), 0.0001f);

            f32 pdf = material->pdf(cos_theta);
            // FIXME: megakernel textures
            vec3 brdf = material->eval(material, rc->textures.get_ptr(), vec2(0.));

            radiance += throughput * emission;
            throughput *= brdf * cos_theta * (1.f / pdf);

            auto rr = russian_roulette(depth, rr_sample, throughput);

            if (!rr.has_value()) {
                return radiance;
            } else {
                auto roulette_compensation = rr.value();
                throughput *= 1.f / roulette_compensation;
            }

            Ray new_ray = spawn_ray(its, sample_dir);
            ray = new_ray;
            depth++;
        } else {
            // Ray has escaped the scene
            if (!rc->has_envmap) {
                return vec3(0.);
            } else {
                const Envmap *envmap = &rc->envmap;
                vec3 envrad = envmap->sample(ray);
                radiance += throughput * envrad;
                return radiance;
            }
        }
    }
}

/// The "megakernel" approach to path-tracing on the GPU
__global__ void render_megakernel(RenderContext *rc) {
    u32 pixel_index = rc->fb.pixel_index(blockDim, blockIdx, threadIdx);

    if (pixel_index < rc->fb.num_pixels()) {
        auto [x, y] = rc->fb.pixel_coords(blockDim, blockIdx, threadIdx);

        vec3 radiance = render(rc, x, y);
        rc->fb.get_pixels()[pixel_index] += radiance;
    }
}
