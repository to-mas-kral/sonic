#ifndef GUISTATE_H
#define GUISTATE_H

#include <SDL2/SDL_opengl.h>

#include <map>
#include <string>
#include <vector>

struct ViewportSettings {
    std::vector<std::string> viewport_targets = {"Main Output", "Denoised"};
    // Indexes into aovs in Viewport
    std::string selected_target{"Main Output"};
};

struct Viewport {
    bool are_textures_init{false};
    GLuint main_output_texture{0};
    // Indexed by m_selected_target in ViewportSettings
    std::map<std::string_view, GLuint> aov_textures;

    GLuint
    select_texture(const std::string_view name) const {
        if (name == "Main Output" || name == "Denoised") {
            return main_output_texture;
        } else {
            return aov_textures.at(name);
        }
    }
};

struct RenderProgress {
    std::atomic_uint32_t samples_done{0};
    // This only applies to the path guiding integrator
    std::optional<std::atomic_uint32_t> current_iteration_max;
    std::atomic_uint32_t current_iteration_done{0};
};

struct GuiState {
    ViewportSettings viewport_settings;
    Viewport viewport;
    RenderProgress render_progress;
};

#endif // GUISTATE_H
