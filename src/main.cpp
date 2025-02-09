#include "embree_accel.h"
#include "gui/gui.h"
#include "integrator/integrator_type.h"
#include "io/image_writer.h"
#include "io/progress_bar.h"
#include "pbrt_loader/pbrt_loader.h"
#include "renderer.h"
#include "settings.h"
#include "utils/basic_types.h"
#include "utils/thread_pool.h"

#include <bit>
#include <chrono>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

namespace sonic {
void
render_headless(Settings settings, Renderer &renderer) {
    spdlog::info("Rendering a {}x{} image at {} spp.", renderer.scene().attribs.film.resx,
                 renderer.scene().attribs.film.resy, settings.spp);

    ProgressBar pb{};
    const auto start{std::chrono::steady_clock::now()};
    for (u32 sample = 1; sample <= settings.spp; sample++) {
        renderer.compute_current_sample();

        const auto end{std::chrono::steady_clock::now()};
        const std::chrono::duration<f64> elapsed{end - start};

        // Update the framebuffer when the number of samples doubles...
        if (std::popcount(renderer.framebuf().num_samples) == 1) {
            ImageWriter::write_framebuffer(settings.out_filename, renderer.framebuf());

            if (settings.save_progress) {
                ImageWriter::write_framebuffer(
                    fmt::format("{}-{}.exr", settings.out_filename,
                                renderer.framebuf().num_samples),
                    renderer.framebuf());
            }
        }

        if (!settings.silent) {
            pb.print(sample, settings.spp, elapsed);
        }

        renderer.reset_iteration_sample();
    }

    ImageWriter::write_framebuffer(settings.out_filename, renderer.framebuf());
}
} // namespace sonic

int
main(int argc, char **argv) {
    /*
     * Parse cmdline arguments
     * */
    Settings settings{};

    CLI::App app{"Sonic, a path-tracer by Tomáš Král, 2023-2024."};
    // argv = app.ensure_utf8(argv);

    const std::map<std::string, IntegratorType> map{
        {"naive", IntegratorType::Naive},
        {"nee", IntegratorType::MISNEE},
        {"pg", IntegratorType::PathGuiding},
        {"nee-lg", IntegratorType::LambdaGuiding},
    };

    app.add_option("--samples", settings.spp, "Samples per pixel (SPP).");
    app.add_option("-s,--scene", settings.scene_path, "Path to the scene file.");
    app.add_option("-o,--out", settings.out_filename,
                   "Path of the output file, without '.exr'");
    app.add_flag("--silent,!--no-silent", settings.silent, "Silent run.")
        ->default_val(true);
    app.add_option("-i,--integrator", settings.integrator_type, "Integrator")
        ->transform(CLI::CheckedTransformer(map, CLI::ignore_case))
        ->default_val(IntegratorType::MISNEE);

    app.add_option("--num-threads", settings.num_threads, "Number of rendering threads")
        ->check(CLI::Range(1, 1024));

    app.add_option("--start-frame", settings.start_frame,
                   "Frame at which to start rendering. Useful for debugging");
    app.add_flag("--load-only", settings.load_only, "Only load the scene.");
    app.add_flag("--no-gui", settings.no_gui, "Run in headless mode.");
    app.add_flag("--save-progress", settings.save_progress, "Save progress images.");

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
        if (settings.scene_path.ends_with(".pbrt")) {
            auto scene_loader = PbrtLoader(std::filesystem::path(settings.scene_path));
            scene_loader.load_scene(scene);
        } else {
            spdlog::error("Unknown scene file format");
            return 1;
        }
    } catch (const std::exception &e) {
        spdlog::error("Error while loading the scene {}", e.what());
        return 1;
    }

    if (settings.load_only) {
        return 0;
    }

    if (settings.out_filename.empty()) {
        settings.out_filename = scene.attribs.film.filename;
    }

    auto renderer = Renderer::init(std::move(scene), settings);

    if (settings.no_gui) {
        sonic::render_headless(settings, renderer);
    } else {
        auto gui = Gui(settings, &renderer);
        gui.run_loop();
    }

    return 0;
}
