#include "embree_device.h"
#include "integrator/integrator.h"
#include "integrator/integrator_type.h"
#include "io/image_writer.h"
#include "io/progress_bar.h"
#include "pbrt_loader/pbrt_loader.h"
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
    std::string out_filename;

    CLI::App app{"A path-tracer by Tomáš Král, 2023-2024."};
    // argv = app.ensure_utf8(argv);

    const std::map<std::string, IntegratorType> map{
        {"naive", IntegratorType::Naive},
        {"mis_nee", IntegratorType::MISNEE},
    };

    app.add_option("--samples", settings.spp, "Samples per pixel (SPP).");
    app.add_option("-s,--scene", scene_path, "Path to the scene file.");
    app.add_option("-o,--out", out_filename, "Path of the output file, without '.exr'");
    app.add_flag("--silent,!--no-silent", settings.silent, "Silent run.")
        ->default_val(true);
    app.add_flag("--normals", settings.render_normals, "Render normals AOV")
        ->default_val(false);
    app.add_option("-i,--integrator", settings.integrator_type, "Integrator")
        ->transform(CLI::CheckedTransformer(map, CLI::ignore_case))
        ->default_val(IntegratorType::MISNEE);

    app.add_option("--num-threads", settings.num_threads, "Number of rendering threads")
        ->check(CLI::Range(1, 1024));

    app.add_option("--start-frame", settings.start_frame,
                   "Frame at which to start rendering. Useful for debugging");
    app.add_flag("--load-only", settings.load_only, "Only load the scene.");

    CLI11_PARSE(app, argc, argv)

    spdlog::set_level(spdlog::level::info);

    if (settings.silent) {
        spdlog::set_level(spdlog::level::err);
    }

    /*
     * Load scene attribs from the scene file
     * */

    Scene scene{};

    spdlog::info("Loading the scene");
    try {
        if (scene_path.ends_with(".pbrt")) {
            auto scene_loader = PbrtLoader(std::filesystem::path(scene_path));
            scene_loader.load_scene(scene);
        } else {
            spdlog::error("Unknown scene file format");
            return 1;
        }
    } catch (const std::exception &e) {
        spdlog::error("Error while loading the scene {}", e.what());
        return 1;
    }

    if (out_filename.empty()) {
        out_filename = scene.attribs.film.filename;
    }

    RenderContext rc(std::move(scene));

    if (settings.load_only) {
        return 0;
    }

    spdlog::info("Creating Embree acceleration structure");
    auto embree_device = EmbreeDevice(rc.scene);

    Integrator integrator(settings, &rc, &embree_device);

    RenderThreads render_threads(rc.attribs, &integrator, settings);

    spdlog::info("Rendering a {}x{} image at {} spp.", rc.attribs.film.resx,
                 rc.attribs.film.resy, settings.spp);

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

        integrator.sample += 1;
        if (!settings.silent) {
            pb.print(s, settings.spp, elapsed);
        }
    }

    render_threads.stop();

    ImageWriter::write_framebuffer(out_filename, rc.fb, settings.spp);

    return 0;
}
