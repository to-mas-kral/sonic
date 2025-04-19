#ifndef GUI_H
#define GUI_H

#include "../renderer.h"
#include "../settings.h"
#include "gui_state.h"

#include <SDL2/SDL.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl2.h>
#include <implot.h>

class Gui {
public:
    Gui(const Settings &settings, Renderer *const renderer)
        : settings(settings), renderer(renderer) {}

    void
    run_loop();

    Gui(const Gui &other) = delete;
    Gui(Gui &&other) noexcept = delete;
    Gui &
    operator=(const Gui &other) = delete;
    Gui &
    operator=(Gui &&other) noexcept = delete;

    ~Gui() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();

        SDL_DestroyWindow(window);
        SDL_GL_DeleteContext(gl_context);
        SDL_Quit();
    }

private:
    void
    viewport_window() const;

    void
    update_gui_state();

    void
    update_viewport_textures();

    void
    process_main_output_color();

    void
    viewport_settings_window();

    void
    render_progress_window() const;

    void
    render_guiding_tree_window();

    void
    render_scene_inspector();

    void
    pokus_window();
    
    void
    main_window();

    static void
    set_imgui_style() {
        ImGui::StyleColorsLight();
        auto &style = ImGui::GetStyle();
        style.ScaleAllSizes(1.2F);
        style.WindowMenuButtonPosition = ImGuiDir_None;
    }

    void
    start_render_thread();

    void
    setup();

    bool
    handle_sdl_event(const SDL_Event &event);

    GuiState gui_state;

    Settings settings;
    Renderer *renderer;

    std::optional<std::jthread> render_thread;
    std::atomic_bool gui_state_needs_update{false};
    std::barrier<> gui_state_update_barrier{2};

    void *gl_context{nullptr};
    SDL_Window *window{nullptr};
    bool is_dockinng_init{false};
};

#endif // GUI_H
