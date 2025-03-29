#include "gui.h"

#include "../math/samplers/halton_sampler.h"

#include <SDL2/SDL_opengl.h>
#include <imgui_internal.h>

void
Gui::run_loop() {
    setup();
    start_render_thread();

    auto done = false;
    while (!done) {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell
        // if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to
        // your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data
        // to your main application, or clear/overwrite your copy of the keyboard
        // data. Generally you may always pass all inputs to dear imgui, and hide them
        // from your application based on those two flags.
        SDL_Event event;

        done = handle_sdl_event(event);

        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) {
                done = true;
            }
            if (event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_CLOSE &&
                event.window.windowID == SDL_GetWindowID(window)) {
                done = true;
            }
        }

        if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED) {
            SDL_Delay(10);
            continue;
        }

        ImGui_ImplSDL2_ProcessEvent(&event);

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        main_window();

        // Rendering
        // (Your code clears your framebuffer, renders your other stuff etc.)
        ImGui::Render();
        const auto &io = ImGui::GetIO();
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        glClearColor(0.F, 0.F, 0.F, 0.F);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }

    render_thread->request_stop();
    const auto _ = gui_state_update_barrier.arrive();
    render_thread->join();
}

void
Gui::update_gui_state() {
    update_viewport_textures();

    if (gui_state_needs_update) {
        const auto &sd_tree = renderer->integrator()->get_sd_tree();
        if (sd_tree.has_value()) {
            gui_state.sd_tree.nodes.clear();

            const auto &tree = sd_tree.value();
            for (int n = 0; n < tree.get_nodes().size(); ++n) {
                const auto &node = tree.get_nodes()[n];

                if (node.is_leaf()) {
                    for (const auto &[mat_id, tree] :
                         node.m_sampling_binarytrees->trees) {
                        std::vector<f32> pdf_x;
                        std::vector<f32> pdf_y;

                        pdf_x.reserve(LAMBDA_SAMPLES);
                        pdf_y.reserve(LAMBDA_SAMPLES);

                        // Update node pdf
                        for (int i = 0; i < LAMBDA_SAMPLES; ++i) {
                            const auto x = LAMBDA_MIN + i * LAMBDA_STEP;
                            const auto y = tree.pdf(x);
                            pdf_x.push_back(x);
                            pdf_y.push_back(y);
                        }

                        auto samples = std::vector<f32>();
                        samples.reserve(1000 * N_SPECTRUM_SAMPLES);
                        for (int i = 0; i < 1000; ++i) {
                            auto sampler = Sampler(uvec2(10, 10), uvec2(1000, 1000), i);
                            auto lambdas = tree.sample(sampler.sample());
                            for (int j = 0; j < N_SPECTRUM_SAMPLES; ++j) {
                                samples.push_back(lambdas[j]);
                            }
                        }

                        gui_state.sd_tree.nodes.push_back(
                            {fmt::format("Node {} - Mat {}", n, mat_id.inner),
                             std::move(pdf_x), std::move(pdf_y), std::move(samples)});
                    }
                }
            }
        }

        const auto &lg_tree = renderer->integrator()->get_lg_tree();
        if (lg_tree.has_value()) {
            gui_state.sd_tree.nodes.clear();

            const auto &tree = lg_tree.value();

            for (u32 mat_id = 0; mat_id < tree.reservoirs().size(); ++mat_id) {
                const auto &reservoirs = tree.reservoirs()[mat_id];

                if (reservoirs == nullptr) {
                    continue;
                }

                for (u32 i = 0; i < reservoirs->reservoirs.size(); ++i) {
                    const auto &reservoir = reservoirs->reservoirs[i];

                    std::vector<f32> pdf_x;
                    std::vector<f32> pdf_y;

                    pdf_x.reserve(LAMBDA_SAMPLES);
                    pdf_y.reserve(LAMBDA_SAMPLES);

                    // Update node pdf
                    for (int j = 0; j < LAMBDA_SAMPLES; ++j) {
                        const auto x = LAMBDA_MIN + j * LAMBDA_STEP;
                        const auto y = reservoir.sampling_binary_tree->pdf(x);
                        pdf_x.push_back(x);
                        pdf_y.push_back(y);
                    }

                    auto samples = std::vector<f32>();
                    samples.reserve(1000 * N_SPECTRUM_SAMPLES);
                    for (int k = 0; k < 1000; ++k) {
                        auto sampler = Sampler(uvec2(10, 10), uvec2(1000, 1000), k);
                        auto lambdas =
                            reservoir.sampling_binary_tree->sample(sampler.sample());
                        for (int j = 0; j < N_SPECTRUM_SAMPLES; ++j) {
                            samples.push_back(lambdas[j]);
                        }
                    }

                    gui_state.sd_tree.nodes.push_back(
                        {fmt::format("Mat {} Reservoir {}", mat_id, i), std::move(pdf_x),
                         std::move(pdf_y), std::move(samples)});
                }
            }
        }
    }

    if (gui_state_needs_update) {
        gui_state_needs_update = false;
        gui_state_update_barrier.arrive_and_wait();
    }
}

void
Gui::update_viewport_textures() {
    const auto width_x = renderer->framebuf().width_x();
    const auto height_y = renderer->framebuf().height_y();

    // The number of AOV textures isn't generally known, so create them
    // dynamically.
    if (!gui_state.viewport.are_textures_init) {
        gui_state.viewport.are_textures_init = true;
        glGenTextures(1, &gui_state.viewport.main_output_texture);
        glBindTexture(GL_TEXTURE_2D, gui_state.viewport.main_output_texture);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    if (gui_state_needs_update) {
        for (auto &[name, aov] : renderer->framebuf().aovs()) {
            if (!gui_state.viewport.aov_textures.contains(name)) {
                GLuint image_texture;
                glGenTextures(1, &image_texture);
                glBindTexture(GL_TEXTURE_2D, image_texture);

                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                gui_state.viewport.aov_textures.insert({name, image_texture});
            }
        }

        // Update textures from framebuf

        glBindTexture(GL_TEXTURE_2D, gui_state.viewport.main_output_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width_x, height_y, 0, GL_RGB, GL_FLOAT,
                     renderer->framebuf().get_pixels().data());

        for (auto &[name, aov] : renderer->framebuf().aovs()) {
            GLint internal_format;
            GLenum data_type;
            GLenum format;
            void *tex_data;
            if (const auto *pixels = std::get_if<std::vector<u32>>(&aov)) {
                internal_format = GL_R32UI;
                data_type = GL_UNSIGNED_INT;
                format = GL_RED_INTEGER;
                tex_data = (void *)pixels->data();
            } else if (const auto *pixels = std::get_if<std::vector<f32>>(&aov)) {
                internal_format = GL_R32F;
                data_type = GL_FLOAT;
                format = GL_RED;
                tex_data = (void *)pixels->data();
            } else if (const auto *pixels = std::get_if<std::vector<tuple3>>(&aov)) {
                internal_format = GL_RGB32F;
                data_type = GL_FLOAT;
                format = GL_RGB;
                tex_data = (void *)pixels->data();
            } else {
                panic("AOV holds unimplemented alternative");
            }

            glBindTexture(GL_TEXTURE_2D, gui_state.viewport.select_texture(name));
            glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width_x, height_y, 0, format,
                         data_type, tex_data);
        }

        for (auto &[name, aov] : renderer->framebuf().aovs()) {
            auto &targets = gui_state.viewport_settings.viewport_targets;
            auto contains = false;
            if (std::find(targets.begin(), targets.end(), name) == targets.end()) {
                contains = true;
            }

            if (contains) {
                targets.push_back(std::string(name));
            }
        }
    }
}

void
Gui::viewport_window() const {
    const auto width_x = renderer->framebuf().width_x();
    const auto height_y = renderer->framebuf().height_y();

    ImGui::Begin("Viewport");
    {
        const auto selected_texture = gui_state.viewport.select_texture(
            gui_state.viewport_settings.selected_target);
        ImGui::Image((ImTextureID)(intptr_t)selected_texture,
                     {(f32)width_x, (f32)height_y});
    }
    ImGui::End();
}

void
Gui::viewport_settings_window() {
    ImGui::Begin("Viewport Settings");
    {
        auto &settings = gui_state.viewport_settings;
        if (ImGui::BeginCombo("Viewport target", settings.selected_target.c_str())) {
            for (int i = 0; i < settings.viewport_targets.size(); i++) {
                const bool is_selected =
                    settings.selected_target == settings.viewport_targets[i];
                if (ImGui::Selectable(settings.viewport_targets[i].c_str(),
                                      is_selected)) {
                    settings.selected_target = settings.viewport_targets[i];

                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
            }

            ImGui::EndCombo();
        }
        ImGui::End();
    }
}

void
Gui::render_progress_window() const {
    ImGui::Begin("Render Progress");
    {
        const auto &rprogress = gui_state.render_progress;

        auto progress_bar = [this](u32 start, u32 end, const char *const label) {
            if (end == 0) {
                return;
            }

            f32 progress = static_cast<f32>(start) / static_cast<f32>(end);
            if (progress > 1.F) {
                progress = 1.F;
            }

            const auto inner_text = fmt::format("{}/{}", start, end);
            ImGui::ProgressBar(progress, ImVec2(0.F, 0.F), inner_text.c_str());
            ImGui::SameLine(0.0F, ImGui::GetStyle().ItemInnerSpacing.x);
            ImGui::Text(label);
        };

        progress_bar(rprogress.samples_done, settings.spp, "Samples");

        if (rprogress.current_iteration_max.has_value()) {
            progress_bar(rprogress.current_iteration_done,
                         rprogress.current_iteration_max.value(), "Iteration samples");
        }
    }
    ImGui::End();
}

void
Gui::render_guiding_tree_window() {
    ImGui::Begin("GuidingTree");

    auto &sd_tree_info = gui_state.sd_tree;

    ImGui::SeparatorText("Leaf nodes");
    const auto num_items = std::min(sd_tree_info.nodes.size(), 8UL);
    if (ImGui::BeginListBox(
            "##", ImVec2(-FLT_MIN, num_items * ImGui::GetTextLineHeightWithSpacing()))) {
        for (int i = 0; i < sd_tree_info.nodes.size(); ++i) {
            const bool is_selected = sd_tree_info.selected_node == i;

            if (ImGui::Selectable(sd_tree_info.nodes[i].name.c_str(), is_selected)) {
                sd_tree_info.selected_node = i;
            }

            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }

        ImGui::EndListBox();
    }

    if (sd_tree_info.selected_node.has_value()) {
        const auto &node = sd_tree_info.nodes[sd_tree_info.selected_node.value()];

        static ImPlotAxisFlags xflags = ImPlotAxisFlags_AutoFit;
        static ImPlotAxisFlags yflags =
            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit;

        if (ImPlot::BeginPlot("Node PDF")) {
            static ImPlotHistogramFlags hist_flags = ImPlotHistogramFlags_Density;
            ImPlot::SetupAxes("X", "Y", xflags, yflags);
            ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.5F);
            ImPlot::PlotHistogram("Samples", node.samples.data(), node.samples.size(),
                                  ImPlotBin_Sqrt, 1.0F, ImPlotRange(), hist_flags);
            ImPlot::SetNextLineStyle(IMPLOT_AUTO_COL, 1.5F);
            ImPlot::PlotLine("PDF", node.pdf_x_values.data(), node.pdf_y_values.data(),
                             node.pdf_x_values.size());
            ImPlot::EndPlot();
        }
    }

    ImGui::End();
}

void
Gui::render_scene_inspector() {
    auto &scene_inspector = gui_state.scene_inspector;
    const auto &lights = renderer->scene().lights;

    ImGui::Begin("SceneInspector");
    {
        ImGui::SeparatorText("Lights");
        const auto num_items = std::min(lights.size(), 8UL);
        if (ImGui::BeginListBox(
                "##",
                ImVec2(-FLT_MIN, num_items * ImGui::GetTextLineHeightWithSpacing()))) {

            for (int n = 0; n < lights.size(); n++) {
                const bool is_selected = scene_inspector.selected_light == n;
                if (ImGui::Selectable(std::to_string(n).c_str(), is_selected)) {
                    if (scene_inspector.selected_light != n) {
                        scene_inspector.selected_light = n;
                        const auto &light = lights[n];

                        // Update light SPD
                        for (int i = 0; i < LAMBDA_SAMPLES; ++i) {
                            const auto x = LAMBDA_MIN + i * LAMBDA_STEP;
                            const auto y = light.emission(x);
                            scene_inspector.spd_x_values[i] = x;
                            scene_inspector.spd_y_values[i] = y;

                            scene_inspector.product_y_values[i] =
                                y * SampledLambdas::pdf_visual_importance(x);

                            scene_inspector.visual_y_values[i] =
                                SampledLambdas::pdf_visual_importance(x);
                        }
                    }
                }

                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndListBox();
        }

        if (scene_inspector.selected_light.has_value()) {
            static ImPlotAxisFlags xflags = ImPlotAxisFlags_AutoFit;
            static ImPlotAxisFlags yflags =
                ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit;
            if (ImPlot::BeginPlot("Light SPD")) {
                ImPlot::SetupAxes(nullptr, nullptr, xflags, yflags);
                ImPlot::PlotLine("power", scene_inspector.spd_x_values.data(),
                                 scene_inspector.spd_y_values.data(),
                                 scene_inspector.spd_x_values.size());
                ImPlot::EndPlot();
            }

            if (ImPlot::BeginPlot("Optimal sampling")) {
                ImPlot::SetupAxes(nullptr, nullptr, xflags, yflags);
                ImPlot::PlotLine("power", scene_inspector.spd_x_values.data(),
                                 scene_inspector.product_y_values.data(),
                                 scene_inspector.spd_x_values.size());
                ImPlot::EndPlot();
            }

            if (ImPlot::BeginPlot("Visual sampling")) {
                ImPlot::SetupAxes(nullptr, nullptr, xflags, yflags);
                ImPlot::PlotLine("power", scene_inspector.spd_x_values.data(),
                                 scene_inspector.visual_y_values.data(),
                                 scene_inspector.spd_x_values.size());
                ImPlot::EndPlot();
            }
        }
    }
    ImGui::End();
}

void
Gui::pokus_window() {
    ImGui::Begin("Pokus");
    {
        const auto regen = [this] {
            gui_state.halton_x.clear();
            gui_state.halton_y.clear();

            const auto dim = gui_state.dimension;
            for (int i = 0; i < gui_state.num_points; ++i) {
                const auto x = HaltonSampler::radical_inverse_permuted(dim, i);
                const auto y = HaltonSampler::radical_inverse_permuted(dim + 1, i);
                gui_state.halton_x.push_back(x);
                gui_state.halton_y.push_back(y);
            }
        };

        if (gui_state.halton_x.empty()) {
            regen();
        }

        auto changed = ImGui::SliderInt("Dimension", &gui_state.dimension, 0, 166);
        changed |= ImGui::SliderInt("Num Points", &gui_state.num_points, 16, 1024);

        if (ImPlot::BeginPlot("Halton Plot", {0, 0})) {
            ImPlot::PlotScatter(fmt::format("Halton dimensions {}-{}",
                                            gui_state.dimension, gui_state.dimension + 1)
                                    .c_str(),
                                gui_state.halton_x.data(), gui_state.halton_y.data(),
                                gui_state.halton_x.size());
            ImPlot::EndPlot();
        }

        if (changed) {
            regen();
        }
    }
    ImGui::End();
}

void
Gui::main_window() {
    auto style = ImGui::GetStyle();
    style.WindowMenuButtonPosition = ImGuiDir_None;

    const auto &io = ImGui::GetIO();
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize(io.DisplaySize);

    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoTitleBar;
    window_flags |= ImGuiWindowFlags_NoScrollbar;
    window_flags |= ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoResize;
    window_flags |= ImGuiWindowFlags_NoCollapse;
    window_flags |= ImGuiWindowFlags_NoDecoration;
    window_flags |= ImGuiWindowFlags_NoDocking;
    bool open = true;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0.0f, 0.0f});

    ImGui::Begin("Main Window", &open, window_flags);
    ImGui::PopStyleVar(3);
    {
        auto dockSpaceId = ImGui::GetID("MainDockSpace");
        is_dockinng_init |= ImGui::DockBuilderGetNode(dockSpaceId) != nullptr;

        constexpr auto dockspaceFlags = ImGuiDockNodeFlags_PassthruCentralNode;
        ImGui::DockSpace(dockSpaceId, {0, 0}, dockspaceFlags);

        if (!is_dockinng_init) {
            is_dockinng_init = true;

            ImGui::DockBuilderRemoveNode(dockSpaceId);
            ImGui::DockBuilderAddNode(dockSpaceId,
                                      dockspaceFlags | ImGuiDockNodeFlags_DockSpace);
            ImGui::DockBuilderSetNodeSize(dockSpaceId, io.DisplaySize);

            auto dockIdRight = ImGui::DockBuilderSplitNode(dockSpaceId, ImGuiDir_Right,
                                                           0.25F, nullptr, &dockSpaceId);

            auto dockIdRightLower = ImGui::DockBuilderSplitNode(
                dockIdRight, ImGuiDir_Down, 0.3F, nullptr, &dockIdRight);

            auto dockIdLeft = ImGui::DockBuilderSplitNode(dockSpaceId, ImGuiDir_Left,
                                                          0.25F, nullptr, &dockSpaceId);

            auto dockIdLeftLower = ImGui::DockBuilderSplitNode(
                dockIdLeft, ImGuiDir_Down, 0.25F, nullptr, &dockIdLeft);

            ImGui::DockBuilderDockWindow("Render Progress", dockIdRight);
            ImGui::DockBuilderDockWindow("Viewport Settings", dockIdRightLower);
            ImGui::DockBuilderDockWindow("Viewport", dockSpaceId);
            ImGui::DockBuilderDockWindow("Scene Inspector", dockIdLeft);
            ImGui::DockBuilderDockWindow("Guiding Tree", dockIdLeftLower);
            ImGui::DockBuilderFinish(dockSpaceId);
        }

        update_gui_state();
        render_scene_inspector();
        render_guiding_tree_window();
        render_progress_window();
        viewport_window();
        viewport_settings_window();

        pokus_window();
    }
    ImGui::End(); // Main Window
}

void
Gui::start_render_thread() {
    render_thread = std::jthread([this](std::stop_token stop_token) {
        spdlog::info("Rendering a {}x{} image at {} spp.",
                     renderer->scene().attribs.film.resx,
                     renderer->scene().attribs.film.resy, settings.spp);
        for (u32 sample = 1; sample <= settings.spp; sample++) {
            renderer->compute_current_sample();
            gui_state.render_progress.samples_done = sample;
            const auto iteration_progress = renderer->iter_progress_info();
            if (iteration_progress.has_value()) {
                gui_state.render_progress.current_iteration_max =
                    iteration_progress->samples_max;
                gui_state.render_progress.current_iteration_done =
                    iteration_progress->samples_done;
            }

            // Update the framebuffer when the number of samples doubles...
            if (std::popcount(renderer->framebuf().num_samples) == 1) {
                gui_state_needs_update.store(true);
                gui_state_update_barrier.arrive_and_wait();
            }

            if (stop_token.stop_requested()) {
                return;
            }

            renderer->reset_iteration_sample();
        }

        gui_state_needs_update.store(true);
        gui_state_update_barrier.arrive_and_wait();
    });
}

/// Mostly taken from Dear Imgui examples:
/// https://github.com/ocornut/imgui/blob/master/examples/example_sdl2_opengl3/main.cpp
void
Gui::setup() {
    // Setup SDL
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0) {
        throw std::runtime_error(
            fmt::format("Error while initializing SDL2: {}", SDL_GetError()));
    }

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100 (WebGL 1.0)
    const char *glsl_version = "#version 100";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#elif defined(IMGUI_IMPL_OPENGL_ES3)
    // GL ES 3.0 + GLSL 300 es (WebGL 2.0)
    const char *glsl_version = "#version 300 es";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#elif defined(__APPLE__)
    // GL 3.2 Core + GLSL 150
    const char *glsl_version = "#version 150";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS,
                        SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG); // Always required on Mac
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
#else
    // GL 3.0 + GLSL 130
    const char *glsl_version = "#version 130";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#endif

    // From 2.0.18: Enable native IME.
#ifdef SDL_HINT_IME_SHOW_UI
    SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");
#endif

    // Create window with graphics context
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    constexpr auto window_flags =
        static_cast<SDL_WindowFlags>(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE |
                                     SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_MAXIMIZED);

    window = SDL_CreateWindow("Sonic", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                              1920, 1080, window_flags);

    if (window == nullptr) {
        throw std::runtime_error(
            fmt::format("Error while initializing SDL2: {}", SDL_GetError()));
    }

    gl_context = SDL_GL_CreateContext(window);
    if (gl_context == nullptr) {
        throw std::runtime_error(
            fmt::format("Error while initializing SDL2: {}", SDL_GetError()));
    }

    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    io.Fonts->AddFontFromFileTTF("./font.ttf", 20);

    set_imgui_style();

    // Setup Platform/Renderer backends
    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);
}

bool
Gui::handle_sdl_event(const SDL_Event &event) {
    if (event.type == SDL_WINDOWEVENT) {
        if (event.window.event == SDL_WINDOWEVENT_CLOSE) {
            return true;
        }
    }

    return false;
}