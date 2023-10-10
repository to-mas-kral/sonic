#ifndef PT_MEGAKERNEL_H
#define PT_MEGAKERNEL_H

#include "../camera.h"
#include "../render_context.h"
#include "../utils/rng.h"

__device__ Ray spawn_ray(Intersection &its, vec3 dir) {
    // TODO: more robust floating-point error handling when spawning rays
    vec3 ray_orig = its.pos + (0.001f * its.normal);
    return Ray(ray_orig, dir);
}

/// Randomly selects if a path should be terminated based on its throughput.
/// Roulette is only applied after the first 3 bounces.
/// Returns true if path should be terminated. If not, also returns roulette compensation.
// TODO: use cuda::std::optional when available
__device__ cuda::std::tuple<bool, f32>
russian_roulette(u32 depth, curandState *rand_state, const vec3 &throughput) {
    if (depth > 3) {
        f32 u = rng(rand_state);
        f32 survival_prob =
            1.f - max(glm::max(throughput.x, throughput.y, throughput.z), 0.05f);

        if (u < survival_prob) {
            return {true, 0.f};
        } else {
            f32 roulette_compensation = 1.f - survival_prob;
            return {false, roulette_compensation};
        }
    } else {
        return {false, 1.f};
    }
}

// (s, t) are coords in screen space
__device__ vec3 render(RenderContext *rc, u32 x, u32 y) {
    auto pixel_index = rc->get_framebuffer().pixel_index(x, y);
    curandState *rand_state = &rc->get_framebuffer().get_rand_state()[pixel_index];

    f32 image_x = static_cast<f32>(rc->get_framebuffer().get_image_x() - 1U);
    f32 image_y = static_cast<f32>(rc->get_framebuffer().get_image_y() - 1U);

    // TODO: make some sort of rng / sampler class
    f32 s_offset = rng(rand_state);
    f32 t_offset = rng(rand_state);

    f32 s = (static_cast<f32>(x) + s_offset) / image_x;
    f32 t = (static_cast<f32>(y) + t_offset) / image_y;

    // clang-format off
    const auto world_to_cam =
        glm::mat4{
                  1., 0., 0., 0.,
                  0., 1., 0., 0.,
                  0., 0., -1., 0.,
                  0., -1., 6.8, 1
        };
    // clang-format on

    auto cam_to_world = glm::inverse(world_to_cam);

    Ray ray = rc->get_cam().get_ray(s, t);
    ray.transform_to_world(cam_to_world);

    /*
     * Iterative naive path tracing
     * */
    u32 depth = 1;
    vec3 throughput = vec3(1.f);
    vec3 radiance = vec3(0.f);

    while (true) {
        Intersection its;
        if (rc->traverse_scene(its, ray)) {
            auto material = &rc->get_materials()[its.material_id];

            vec3 emission = vec3(0.f);
            if (its.mesh->has_light()) {
                auto light_id = its.mesh->get_light_id();
                emission = rc->get_lights()[light_id].emission();
            }

            if (glm::dot(-ray.dir, its.normal) < 0.f) {
                its.normal = -its.normal;
                emission = vec3(0.f);
            }

            vec3 sample_dir = material->sample(its.normal, -ray.dir, rand_state);
            Ray new_ray = spawn_ray(its, sample_dir);
            f32 pdf = material->pdf(its.normal, sample_dir);
            vec3 brdf = material->eval();

            f32 cos_theta = glm::dot(its.normal, sample_dir);

            radiance += throughput * emission;
            throughput *= brdf * cos_theta * (1.f / pdf);

            auto [should_terminate, roulette_compensation] =
                russian_roulette(depth, rand_state, throughput);

            if (should_terminate) {
                return radiance;
            } else {
                throughput *= 1.f / roulette_compensation;
            }

            ray = new_ray;
            depth++;
        } else {
            // Could do some environment mapping...
            return vec3(0.);
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

#endif // PT_MEGAKERNEL_H
