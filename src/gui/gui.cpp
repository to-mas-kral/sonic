#include "gui.h"

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
    const auto _ = render_barrier.arrive();
    render_thread->join();
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
    if (render_outputs_need_update) {
        render_outputs_need_update = false;

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

        render_barrier.arrive_and_wait();
    }
}

void
Gui::viewport_window() {
    update_viewport_textures();

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
    for (auto &[name, aov] : renderer->framebuf().aovs()) {
        auto &targets = gui_state.viewport_settings.viewport_targets;
        if (std::find(targets.begin(), targets.end(), name) == targets.end()) {
            targets.push_back(std::string(name));
        }
    }

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

        auto progress_bar = [this](u32 start, u32 end, const char *label) {
            if (end == 0) {
                return;
            }

            f32 progress = static_cast<f32>(start) / static_cast<f32>(end);
            if (progress > 1.F) {
                progress = 1.F;
            }

            const auto inner_text = fmt::format("{}/{}", start, end);
            ImGui::ProgressBar(progress, ImVec2(0.f, 0.f), inner_text.c_str());
            ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
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

            ImGui::DockBuilderDockWindow("Render Progress", dockIdRight);
            ImGui::DockBuilderDockWindow("Viewport Settings", dockIdRightLower);
            ImGui::DockBuilderDockWindow("Viewport", dockSpaceId);
            ImGui::DockBuilderFinish(dockSpaceId);
        }

        render_progress_window();
        viewport_window();
        viewport_settings_window();
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
            renderer->compute_sample();
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
                render_outputs_need_update.store(true);
                render_barrier.arrive_and_wait();
            }

            if (stop_token.stop_requested()) {
                return;
            }
        }

        render_outputs_need_update.store(true);
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
    constexpr auto window_flags = static_cast<SDL_WindowFlags>(
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);

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