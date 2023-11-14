#include <bit>
#include <chrono>

#include <CLI/CLI.hpp>
#include <fmt/core.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_host.h>
#include <optix_stubs.h>
#include <spdlog/spdlog.h>

#include "io/image_writer.h"
#include "io/progress_bar.h"
#include "io/window.h"
#include "kernels/megakernel.h"
#include "kernels/raygen.h"
#include "optix_as.h"
#include "optix_common.h"
#include "optix_renderer.h"
#include "render_context_common.h"
#include "scene_loader.h"
#include "utils/cuda_err.h"
#include "utils/shared_vector.h"

// FIXME: there is a memory error in OptiX sphere acceleration creation, but seems to be
// an issue in Nvidia's code. Try when new CUDA version is released...
//==51801== Conditional jump or move depends on uninitialised value(s)
//==51801==    at 0x261576F9: ??? (in /usr/lib/libnvidia-rtcore.so.545.29.02)
//==51801==    by 0x2613C96D: ??? (in /usr/lib/libnvidia-rtcore.so.545.29.02)
//==51801==    by 0x2614B415: ??? (in /usr/lib/libnvidia-rtcore.so.545.29.02)
//==51801==    by 0x25F3B18D: ??? (in /usr/lib/libnvidia-rtcore.so.545.29.02)
//==51801==    by 0x25F3CCAA: ??? (in /usr/lib/libnvidia-rtcore.so.545.29.02)
//==51801==    by 0x225FA5CB: ??? (in /usr/lib/libnvoptix.so.545.29.02)
//==51801==    by 0x225F82DE: ??? (in /usr/lib/libnvoptix.so.545.29.02)
//==51801==    by 0x128B3F: optixAccelComputeMemoryUsage (optix_stubs.h:489)
//==51801==    by 0x12BFF6: OptixAS::create_as(OptixDeviceContext_t*,
//std::vector<OptixBuildInput, std::allocator<OptixBuildInput> > const&, unsigned long
//long*) (optix_as.h:83)
//==51801==    by 0x12CBC8: OptixAS::OptixAS(RenderContext*, OptixDeviceContext_t*)
//(optix_as.h:175)
//==51801==    by 0x125595: main (main.cu:104)
//==51801==  Uninitialised value was created by a stack allocation
//==51801==    at 0x2614B16B: ??? (in /usr/lib/libnvidia-rtcore.so.545.29.02)

int main(int argc, char **argv) {
    auto optix_context = init_optix();

    // TODO: wrap this in some class... need to have a block so that OptixRenderer
    // destructor is called before resetting the device at the end of main()...
    {
        /*
         * Parse comdline arguments
         * */

        u32 num_samples = 32;
        bool silent = false;
        bool optix = true;
        std::string scene_path{};

        CLI::App app{"A CUDA path-tracer project for PGRF3 by Tomáš Král, 2023."};
        // argv = app.ensure_utf8(argv);

        app.add_option("--samples", num_samples, "Number of samples.");
        app.add_option("-s,--scene", scene_path, "Path to the scene file.");
        app.add_flag("--silent,!--no-silent", silent, "Silent run.")->default_val(true);
        app.add_flag("--optix,!--no-optix", optix, "Use OptiX.")->default_val(true);

        CLI11_PARSE(app, argc, argv);

        spdlog::set_level(spdlog::level::info);

        if (silent) {
            spdlog::set_level(spdlog::level::err);
        }

        /*
         * Load scene attribs from the scene file
         * */

        SceneLoader scene_loader;
        try {
            scene_loader = SceneLoader(scene_path);
        } catch (const std::exception &e) {
            spdlog::error("Error while parsing the scene file");
            return 1;
        };
        auto attrib_result = scene_loader.load_scene_attribs();
        if (!attrib_result.has_value()) {
            spdlog::error("Error while getting scene attribs");
            return 1;
        }
        SceneAttribs attribs = attrib_result.value();

        /*
         * Window setup
         * */
        auto window = Window(attribs.resx, attribs.resy);

        /*
         * Set up render context
         * */

        // TODO: could probably make some template class for this...
        RenderContext *rc;
        CUDA_CHECK(cudaMallocManaged((void **)&rc, sizeof(RenderContext)));
        auto rcx = new (rc) RenderContext(num_samples, attribs);

        /*
         * Load the scene
         * */

        spdlog::info("Loading the scene");
        try {
            scene_loader.load_scene(rc);
        } catch (const std::exception &e) {
            spdlog::error("Error while loading the scene {}", e.what());
            return 1;
        }
        spdlog::info("Scene loaded");

        rc->geometry.fixup_geometry_pointers();

        spdlog::info("Creating OptiX acceleration structure");
        auto optix_as = OptixAS(rc, optix_context);
        spdlog::info("OptiX acceleration structure initialized");
        auto optix_renderer = OptixRenderer(rc, optix_context, &optix_as);

        /*
         * Start rendering
         * */

        dim3 blocks_dim = rc->blocks_dim;
        dim3 threads_dim = rc->THREADS_DIM;

        spdlog::info("Rendering a {}x{} image at {} samples.", attribs.resx, attribs.resy,
                     num_samples);

        if (!optix) {
            spdlog::info("Creating BVH acceleration structure");
            rc->make_acceleration_structure();
            spdlog::info("BVH acceleration structure created");

            spdlog::info("Pixel grid split into {} blocks with {} threads each.",
                         blocks_dim.x * blocks_dim.y, threads_dim.x * threads_dim.y);
        }

        PtParams params{};
        // Pass straight to params due to performance reasons...
        // No need to traverse 1 extra pointer...
        params.rc = rc;
        params.fb = &rc->fb;
        params.meshes = rc->geometry.meshes.meshes.get_ptr();
        params.materials = rc->materials.get_ptr();
        params.lights = rc->lights.get_ptr();
        params.textures = rc->textures.get_ptr();

        ProgressBar pb;

        const auto start{std::chrono::steady_clock::now()};

        // OptiX path-tracer
        for (u32 s = 1; s <= num_samples; s++) {
            params.sample_index = s - 1;
            if (optix) {
                optix_renderer.launch(params, attribs.resx, attribs.resy);
            } else {
                render_megakernel<<<blocks_dim, threads_dim>>>(rc);

                cudaDeviceSynchronize();
                CUDA_CHECK_LAST_ERROR();
            }

            const auto end{std::chrono::steady_clock::now()};
            const std::chrono::duration<f64> elapsed{end - start};

            // Update the framebuffer when the number of samples doubles...
            if (std::popcount(s) == 1) {
                window.update(rc->fb, s);
                ImageWriter::write_framebuffer("ptout.exr", rc->fb, s);
            }

            pb.print(s, num_samples, elapsed);
        }

        // spdlog::info("Shot a total of {} rays", rc->ray_counter.fetch_add(0));

        /*
         * Clean up and exit
         * */

        window.close();

        cudaDeviceSynchronize();

        // Call the destructor manually, so the memory inside of RenderContext
        // deallocates.
        rc->~RenderContext();
        CUDA_CHECK(cudaFree(rc));
        CUDA_CHECK_LAST_ERROR();
    }

    OPTIX_CHECK(optixDeviceContextDestroy(optix_context));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
