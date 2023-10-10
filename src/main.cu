#include <bit>

#include <CLI/CLI.hpp>

#include <curand_kernel.h>
#include <fmt/core.h>

#include "geometry/ray.h"
#include "kernels/megakernel.h"
#include "render_context.h"
#include "shapes/mesh.h"
#include "utils/cuda_err.h"
#include "utils/image_writer.h"
#include "utils/shared_vector.h"

void init_meshes(RenderContext *rc) {
    // Taken from the PBRTv4 version of the Cornell Box by Benedikt Bitterli

    /*
     * Materials
     * */

    //"LeftWall"
    rc->add_material(Material(vec3(0.63, 0.065, 0.05)));
    //"RightWall"
    rc->add_material(Material(vec3(0.14, 0.45, 0.091)));

    //"Rest - light gray"
    rc->add_material(Material(vec3(0.725, 0.71, 0.68)));

    //"Light"
    rc->add_material(Material(vec3(0, 0, 0)));

    /*
     * Lights
     * */
    rc->add_light(Light(vec3(17., 12., 4.)));

    /*
     * Meshes
     * */

    // Floor
    {
        SharedVector<u32> indices{0, 1, 2, 0, 2, 3};
        SharedVector<f32> pos{-1.f, 1.74846e-7f,  -1.f, -1.f, 1.74846e-7f,  1.f,
                              1.f,  -1.74846e-7f, 1.f,  1.f,  -1.74846e-7f, -1.f};

        Mesh mesh(std::move(indices), std::move(pos), 2);
        rc->add_mesh(std::move(mesh));
    }

    // Ceiling
    {
        SharedVector<u32> indices{
            0, 1, 2, 0, 2, 3,
        };
        SharedVector<f32> pos{1.f,  2.f, 1.f,  -1.f, 2.f, 1.f,
                              -1.f, 2.f, -1.f, 1.f,  2.f, -1.f};

        Mesh mesh(std::move(indices), std::move(pos), 2);
        rc->add_mesh(std::move(mesh));
    }

    // Backwall
    {
        SharedVector<u32> indices{
            0, 1, 2, 0, 2, 3,
        };
        SharedVector<f32> pos{-1.f, 0.f, -1.f, -1.f, 2.f, -1.f,
                              1.f,  2.f, -1.f, 1.f,  0.f, -1.f};

        Mesh mesh(std::move(indices), std::move(pos), 2);
        rc->add_mesh(std::move(mesh));
    }

    // LeftWall
    {
        SharedVector<u32> indices{
            0, 1, 2, 0, 2, 3,
        };
        SharedVector<f32> pos{-1.f, 0.f, 1.f,  -1.f, 2.f, 1.f,
                              -1.f, 2.f, -1.f, -1.f, 0.f, -1.f};

        Mesh mesh(std::move(indices), std::move(pos), 0);
        rc->add_mesh(std::move(mesh));
    }

    // RightWall
    {
        SharedVector<u32> indices{
            0, 1, 2, 0, 2, 3,
        };
        SharedVector<f32> pos{1.f, 0.f, -1.f, 1.f, 2.f, -1.f,
                              1.f, 2.f, 1.f,  1.f, 0.f, 1.f};

        Mesh mesh(std::move(indices), std::move(pos), 1);
        rc->add_mesh(std::move(mesh));
    }

    // Tallbox
    {
        SharedVector<u32> indices{0,  2,  1,  0,  3,  2,  4,  6,  5,  4,  7,  6,
                                  8,  10, 9,  8,  11, 10, 12, 14, 13, 12, 15, 14,
                                  16, 18, 17, 16, 19, 18, 20, 22, 21, 20, 23, 22};

        SharedVector<f32> pos{-0.720444, 1.2, -0.473882, -0.720444, 0,   -0.473882,
                              -0.146892, 0,   -0.673479, -0.146892, 1.2, -0.673479,
                              -0.523986, 0,   0.0906493, -0.523986, 1.2, 0.0906492,
                              0.0495656, 1.2, -0.108948, 0.0495656, 0,   -0.108948,
                              -0.523986, 1.2, 0.0906492, -0.720444, 1.2, -0.473882,
                              -0.146892, 1.2, -0.673479, 0.0495656, 1.2, -0.108948,
                              0.0495656, 0,   -0.108948, -0.146892, 0,   -0.673479,
                              -0.720444, 0,   -0.473882, -0.523986, 0,   0.0906493,
                              -0.523986, 0,   0.0906493, -0.720444, 0,   -0.473882,
                              -0.720444, 1.2, -0.473882, -0.523986, 1.2, 0.0906492,
                              0.0495656, 1.2, -0.108948, -0.146892, 1.2, -0.673479,
                              -0.146892, 0,   -0.673479, 0.0495656, 0,   -0.108948};

        Mesh mesh(std::move(indices), std::move(pos), 2);
        rc->add_mesh(std::move(mesh));
    }

    // Shortbox
    {
        SharedVector<u32> indices{0,  2,  1,  0,  3,  2,  4,  6,  5,  4,  7,  6,
                                  8,  10, 9,  8,  11, 10, 12, 14, 13, 12, 15, 14,
                                  16, 18, 17, 16, 19, 18, 20, 22, 21, 20, 23, 22};

        SharedVector<f32> pos{
            -0.0460751, 0.6,         0.573007,   -0.0460751, -2.98023e-8, 0.573007,
            0.124253,   0,           0.00310463, 0.124253,   0.6,         0.00310463,
            0.533009,   0,           0.746079,   0.533009,   0.6,         0.746079,
            0.703337,   0.6,         0.176177,   0.703337,   2.98023e-8,  0.176177,
            0.533009,   0.6,         0.746079,   -0.0460751, 0.6,         0.573007,
            0.124253,   0.6,         0.00310463, 0.703337,   0.6,         0.176177,
            0.703337,   2.98023e-8,  0.176177,   0.124253,   0,           0.00310463,
            -0.0460751, -2.98023e-8, 0.573007,   0.533009,   0,           0.746079,
            0.533009,   0,           0.746079,   -0.0460751, -2.98023e-8, 0.573007,
            -0.0460751, 0.6,         0.573007,   0.533009,   0.6,         0.746079,
            0.703337,   0.6,         0.176177,   0.124253,   0.6,         0.00310463,
            0.124253,   0,           0.00310463, 0.703337,   2.98023e-8,  0.176177};

        Mesh mesh(std::move(indices), std::move(pos), 2);
        rc->add_mesh(std::move(mesh));
    }

    // Light
    {
        SharedVector<f32> pos{-0.24, 1.98, -0.22, 0.23,  1.98, -0.22,
                              0.23,  1.98, 0.16,  -0.24, 1.98, 0.16};

        SharedVector<u32> indices{0, 1, 2, 0, 2, 3};

        Mesh mesh(std::move(indices), std::move(pos), 3, 0);
        rc->add_mesh(std::move(mesh));
    }
}

int main(int argc, char **argv) {
    /*
     * Parse comdline arguments
     * */

    u32 image_x = 1280;
    u32 image_y = 720;
    u32 num_samples = 32;

    CLI::App app{"A CUDA path-tracer project for PGRF3 by Tomáš Král, 2023."};
    // argv = app.ensure_utf8(argv);

    app.add_option("-s,--samples", num_samples, "Number of samples.");
    app.add_option("-x,--width", image_x, "Image width in pixels.");
    app.add_option("-y,--height", image_y, "Image height in pixels.");

    bool silent = false;

    app.add_flag("--silent,!--no-silent", silent, "Silent run.")->default_val(true);

    fmt::println("{}", silent);

    CLI11_PARSE(app, argc, argv);

    /*
     * Set up render context
     * */

    RenderContext *rc;
    CUDA_CHECK(cudaMallocManaged((void **)&rc, sizeof(RenderContext)));
    auto rcx = new (rc) RenderContext(num_samples, image_x, image_y);

    init_meshes(rc);

    /*
     * Start rendering
     * */

    dim3 blocks_dim = rc->get_blocks_dim();
    dim3 threads_dim = rc->get_threads_dim();

    if (!silent) {
        fmt::println("Rendering a {}x{} image at {} samples.", image_x, image_y,
                     num_samples);
        fmt::println("Pixel grid split into {} blocks with {} threads each.",
                     blocks_dim.x * blocks_dim.y, threads_dim.x * threads_dim.y);
    }

    // Moving this for loop into the kernel is a bit faster, but this lets me report
    // progress...
    for (u32 s = 0; s < num_samples; s++) {
        render_megakernel<<<blocks_dim, threads_dim>>>(rc);

        cudaDeviceSynchronize();
        CUDA_CHECK_LAST_ERROR();

        if (!silent) {
            fmt::println("Sample {} done.", s + 1);
        }

        // Update the framebuffer when the number of samples doubles...
        if (std::popcount(s + 1) == 1) {
            if (!silent) {
                fmt::println("Updating framebuffer");
            }
            ImageWriter::write_framebuffer(rc->get_framebuffer(), s + 1);
        }
    }

    /*
     * Clean up and exit
     * */

    cudaDeviceSynchronize();

    // Call the destructor manually, so the memory inside of RenderContext
    // deallocates.
    rc->~RenderContext();
    CUDA_CHECK_LAST_ERROR();

    CUDA_CHECK(cudaFree(rc));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
