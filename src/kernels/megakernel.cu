
#include "megakernel.h"

#include "../geometry/intersection.h"
#include "../integrator/utils.h"
#include "../render_context_common.h"
#include "../utils/rng.h"
#include "raygen.h"

// (s, t) are coords in screen space
__device__ vec3 render(RenderContext *rc, u32 x, u32 y) {
    auto pixel_index = rc->get_framebuffer().pixel_index(x, y);
    curandState *rand_state = &rc->get_framebuffer().get_rand_state()[pixel_index];

    Ray ray = gen_ray(x, y, &rc->get_framebuffer(), rc);

    /*
     * Iterative naive path tracing
     * */
    u32 depth = 1;
    vec3 throughput = vec3(1.f);
    vec3 radiance = vec3(0.f);

    while (true) {
        Intersection its;
        // rc->ray_counter.fetch_add(1);
        if (rc->intersect_scene(its, ray)) {
            auto material = &rc->get_materials()[its.mesh->material_id];

            vec3 emission = vec3(0.f);
            if (its.mesh->has_light()) {
                emission = rc->get_lights()[its.mesh->light_id].emission();
            }

            if (glm::dot(-ray.dir, its.normal) < 0.f) {
                its.normal = -its.normal;
                emission = vec3(0.f);
            }

            vec3 sample_dir = material->sample(its.normal, -ray.dir, rand_state);
            // TODO: what to do when cos_theta is 0 ? this minimum value is a band-aid
            // solution...
            f32 cos_theta = max(glm::dot(its.normal, sample_dir), 0.0001f);

            f32 pdf = material->pdf(cos_theta);
            vec3 brdf = material->eval();

            radiance += throughput * emission;
            throughput *= brdf * cos_theta * (1.f / pdf);

            auto [should_terminate, roulette_compensation] =
                russian_roulette(depth, rand_state, throughput);

            if (should_terminate) {
                return radiance;
            } else {
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
                const Envmap *envmap = rc->get_envmap();
                vec3 envrad = envmap->sample(ray);
                radiance += throughput * envrad;
                return radiance;
            }
        }
    }
}

/// The "megakernel" approach to path-tracing on the GPU
__global__ void render_megakernel(RenderContext *rc) {
    u32 pixel_index = rc->get_framebuffer().pixel_index(blockDim, blockIdx, threadIdx);

    if (pixel_index < rc->get_framebuffer().num_pixels()) {
        auto [x, y] = rc->get_framebuffer().pixel_coords(blockDim, blockIdx, threadIdx);

        vec3 radiance = render(rc, x, y);
        rc->get_framebuffer().get_pixels()[pixel_index] += radiance;
    }
}
