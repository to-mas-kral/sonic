#ifndef GUI_H
#define GUI_H

#include "../renderer.h"
#include "../settings.h"
#include "gui_state.h"

#include <SDL2/SDL.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl2.h>

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
        ImGui::DestroyContext();

        SDL_DestroyWindow(window);
        SDL_GL_DeleteContext(gl_context);
        SDL_Quit();
    }

private:
    void
    viewport_window();

    void
    update_viewport_textures();

    void
    viewport_settings_window();

    void
    render_progress_window() const;

    void
    main_window();

    static void
    set_imgui_style() {
        ImGui::StyleColorsLight();
        auto &style = ImGui::GetStyle();
        style.ScaleAllSizes(2.5F);
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
    std::atomic_bool render_outputs_need_update{false};
    std::barrier<> render_barrier{2};

    void *gl_context{nullptr};
    SDL_Window *window{nullptr};
    bool is_dockinng_init{false};
};

#endif // GUI_H
