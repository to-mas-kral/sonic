#include <bit>

#include <CLI/CLI.hpp>
#include <fmt/core.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_host.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include "kernels/megakernel.h"
#include "kernels/optix_triangle.h"
#include "kernels/raygen.h"
#include "kernels/wavefront_common.h"
#include "render_context_common.h"
#include "scene_loader.h"
#include "utils/cuda_err.h"
#include "utils/image_writer.h"
#include "utils/shared_vector.h"

auto read_file(std::string_view path) -> std::string {
    constexpr auto read_size = std::size_t(4096);
    auto stream = std::ifstream(path.data(), std::ios::binary);
    stream.exceptions(std::ios_base::badbit);

    if (not stream) {
        throw std::ios_base::failure("file does not exist");
    }

    auto out = std::string();
    auto buf = std::string(read_size, '\0');
    while (stream.read(&buf[0], read_size)) {
        out.append(buf, 0, stream.gcount());
    }
    out.append(buf, 0, stream.gcount());
    return out;
}

int main(int argc, char **argv) {
    /*
     * Parse comdline arguments
     * */

    u32 num_samples = 32;
    bool silent = false;
    std::string scene_path{};

    CLI::App app{"A CUDA path-tracer project for PGRF3 by Tomáš Král, 2023."};
    // argv = app.ensure_utf8(argv);

    app.add_option("--samples", num_samples, "Number of samples.");
    app.add_option("-s,--scene", scene_path, "Path to the scene file.");
    app.add_flag("--silent,!--no-silent", silent, "Silent run.")->default_val(true);

    CLI11_PARSE(app, argc, argv);

    /*
     * Load scene attribs from the scene file
     * */

    SceneLoader scene_loader;
    try {
        scene_loader = SceneLoader(scene_path);
    } catch (const std::exception &e) {
        fmt::println("Error while parsing the scene file");
        return 1;
    };
    auto attrib_result = scene_loader.load_scene_attribs();
    if (!attrib_result.has_value()) {
        fmt::println("Error while getting scene attribs");
        return 1;
    }
    SceneAttribs attribs = attrib_result.value();

    /*
     * Set up render context and wavefront state
     * */

    // TODO: could probably make some template class for this...
    RenderContext *rc;
    CUDA_CHECK(cudaMallocManaged((void **)&rc, sizeof(RenderContext)));
    auto rcx = new (rc) RenderContext(num_samples, attribs);

    WavefrontState *ws;
    CUDA_CHECK(cudaMallocManaged((void **)&ws, sizeof(WavefrontState)));
    auto wsx = new (ws) WavefrontState(attribs);

    /*
     * Load the scene
     * */

    try {
        scene_loader.load_scene(rc);
    } catch (const std::exception &e) {
        fmt::println("Error while loading the scene");
        return 1;
    }

    rc->make_acceleration_structure();

    /*
     * Start rendering
     * */

    dim3 blocks_dim = rc->get_blocks_dim();
    dim3 threads_dim = rc->get_threads_dim();

    if (!silent) {
        fmt::println("Rendering a {}x{} image at {} samples.", attribs.resx, attribs.resy,
                     num_samples);
        fmt::println("Pixel grid split into {} blocks with {} threads each.",
                     blocks_dim.x * blocks_dim.y, threads_dim.x * threads_dim.y);
    }

    // Wavefront approach
    for (u32 s = 0; s < num_samples; s++) {
        raygen<<<blocks_dim, threads_dim>>>(rc, ws);

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
            ImageWriter::write_framebuffer("ptout.exr", rc->get_framebuffer(), s + 1);
        }
    }

    // Megakernel approach
    /*for (u32 s = 0; s < num_samples; s++) {
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
            ImageWriter::write_framebuffer("ptout.exr", rc->get_framebuffer(), s + 1);
        }
    }*/

    fmt::println("Shot a total of {} rays", rc->ray_counter.fetch_add(0));

    /*
     * Clean up and exit
     * */

    cudaDeviceSynchronize();

    // Call the destructor manually, so the memory inside of RenderContext
    // deallocates.
    rc->~RenderContext();
    CUDA_CHECK(cudaFree(rc));
    CUDA_CHECK(cudaFree(ws));
    CUDA_CHECK_LAST_ERROR();

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
