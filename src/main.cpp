#include "embree_device.h"
#include "integrator/integrator.h"
#include "integrator/integrator_type.h"
#include "io/image_writer.h"
#include "io/progress_bar.h"
#include "io/scene_loader.h"
#include "render_context.h"
#include "settings.h"
#include "utils/basic_types.h"
#include "utils/render_threads.h"

#include <bit>
#include <chrono>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

int
main(int argc, char **argv) {
    /*
     * Parse comdline arguments
     * */

    Settings settings{};
    std::string scene_path{};
    std::string out_filename = "out";

    CLI::App app{"A path-tracer by Tomáš Král, 2023-2024."};
    // argv = app.ensure_utf8(argv);

    std::map<std::string, IntegratorType> map{
        {"naive", IntegratorType::Naive},
        {"mis_nee", IntegratorType::MISNEE},
    };

    app.add_option("--samples", settings.spp, "Samples per pixel (SPP).");
    app.add_option("-s,--scene", scene_path, "Path to the scene file.");
    app.add_option("-o,--out", out_filename, "Path of the output file, without '.exr'");
    app.add_flag("--silent,!--no-silent", settings.silent, "Silent run.")
        ->default_val(true);
    app.add_flag("--save-tx-separate", settings.save_tx_separate,
                 "Save separate images for paths of increasing length")
        ->default_val(false);
    app.add_option("-i,--integrator", settings.integrator_type, "Integrator")
        ->transform(CLI::CheckedTransformer(map, CLI::ignore_case))
        ->default_val(IntegratorType::MISNEE);

    app.add_option("--start-frame", settings.start_frame,
                   "Frame at which to start rendering. Useful for debugging");

    CLI11_PARSE(app, argc, argv)

    spdlog::set_level(spdlog::level::info);

    if (settings.silent) {
        spdlog::set_level(spdlog::level::err);
    }

    /*
     * Load scene attribs from the scene file
     * */

    SceneLoader scene_loader;
    try {
        scene_loader = SceneLoader(scene_path);
    } catch (const std::exception &e) {
        spdlog::error("Error while loading the scene: {}", e.what());
        return 1;
    }
    auto attrib_result = scene_loader.load_scene_attribs();
    if (!attrib_result.has_value()) {
        spdlog::error("Error while getting scene attribs");
        return 1;
    }
    SceneAttribs attribs = attrib_result.value();

    RenderContext rc(attribs);

    spdlog::info("Loading the scene");
    try {
        scene_loader.load_scene(rc.scene);
    } catch (const std::exception &e) {
        spdlog::error("Error while loading the scene {}", e.what());
        return 1;
    }

    rc.scene.init_light_sampler();

    spdlog::info("Creating Embree acceleration structure");
    auto embree_device = EmbreeDevice(rc.scene);

    Integrator integrator(settings, &rc, &embree_device);

    RenderThreads render_threads(rc.attribs, &integrator);

    spdlog::info("Rendering a {}x{} image at {} spp.", attribs.camera_attribs.resx,
                 attribs.camera_attribs.resy, settings.spp);

    ProgressBar pb{};
    const auto start{std::chrono::steady_clock::now()};

    for (u32 s = 1; s <= settings.spp; s++) {
        render_threads.start_new_frame();

        const auto end{std::chrono::steady_clock::now()};
        const std::chrono::duration<f64> elapsed{end - start};

        // Update the framebuffer when the number of samples doubles...
        if (std::popcount(s) == 1) {
            ImageWriter::write_framebuffer(out_filename, rc.fb, s);
        }

        integrator.frame += 1;
        pb.print(s, settings.spp, elapsed);
    }

    render_threads.stop();

    ImageWriter::write_framebuffer(out_filename, rc.fb, settings.spp);

    return 0;
}
