#include <cfloat>
#include <ctime>
#include <iostream>

#include <curand_kernel.h>
#include <fmt/core.h>

#include "camera.h"
#include "ray.h"

#include "kernels/megakernel.h"
#include "render_context.h"
#include "shapes/mesh.h"
#include "utils/cuda_err.h"
#include "utils/image_writer.h"
#include "utils/shared_vector.h"

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
/*__device__ vec3 color(const ray &r, hitable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered,
                                     local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return vec3(0.0, 0.0, 0.0);
            }
        } else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam,
                       hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

SharedVector<Mesh> load_meshes();

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera,
                             int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] =
            new sphere(vec3(0, -1000.0, -1), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) {
                    d_list[i++] =
                        new sphere(center, 0.2,
                                   new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                } else if (choose_mat < 0.95f) {
                    d_list[i++] = new sphere(
                        center, 0.2,
                        new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND),
                                       0.5f * (1.0f + RND)),
                                  0.5f * RND));
                } else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] =
            new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, 22 * 22 + 1 + 3);

        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0;
        (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(lookfrom, lookat, vec3(0, 1, 0), 30.0,
                               float(nx) / float(ny), aperture, dist_to_focus);
    }
} */

void init_meshes(RenderContext *rc) {
    u32 material_id = 0;

    SharedVector<u32> indices{0, 1, 2, 0, 2, 3};
    SharedVector<f32> pos{-1.f, 1.74846e-7f,  -1.f, -1.f, 1.74846e-7f,  1.f,
                          1.f,  -1.74846e-7f, 1.f,  1.f,  -1.74846e-7f, -1.f};

    Mesh mesh0(std::move(indices), std::move(pos), material_id);
    rc->add_mesh(std::move(mesh0));
}

int main() {
    // TODO: get these from cmdline
    u32 image_x = 800;
    u32 image_y = 600;
    u32 num_samples = 256;

    auto fb = Framebuffer(image_x, image_y);

    RenderContext *rc;
    CUDA_CHECK(cudaMallocManaged((void **)&rc, sizeof(RenderContext)));
    *rc = RenderContext(num_samples, image_x, image_y);

    init_meshes(rc);

    u32 num_blocks = rc->get_blocks();
    u32 threads_per_block = rc->get_threads_per_block();

    fmt::println("Rendering a {}x{} image at {} samples.", image_x, image_y, num_samples);
    fmt::println("{} samples split into {} blocks with {} samples each.", num_samples,
                 num_blocks, threads_per_block);

    for (u32 x = 0; x < image_x; x++) {
        for (u32 y = 0; y < image_y; y++) {
            render_megakernel<<<num_blocks, threads_per_block>>>(rc, x, y);
            cudaDeviceSynchronize();

            auto &sample_accum = rc->get_sample_accum();
            auto accum = vec3(0.f, 0.f, 0.f);
            for (int i = 0; i < sample_accum.len(); i++) {
                accum += sample_accum[i];
            }

            fb.get_pixels()[fb.pixel_index(x, y)] = accum / static_cast<f32>(num_samples);
        }
    }

    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR();

    ImageWriter::write_framebuffer(fb);

    // Call the destructor manually, so the memory inside of RenderContext deallocates.
    rc->~RenderContext();
    CUDA_CHECK_LAST_ERROR();

    CUDA_CHECK(cudaFree(rc));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
