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

constexpr f32 LAMBDA_SAMPLES = 1000;
constexpr f32 LAMBDA_STEP = (LAMBDA_MAX - LAMBDA_MIN) / LAMBDA_SAMPLES;

struct SceneInspector {
    SceneInspector()
        : spd_x_values(LAMBDA_SAMPLES, 0.F), spd_y_values(LAMBDA_SAMPLES, 0.F) {}

    std::optional<u32> selected_light;
    std::vector<f32> spd_x_values;
    std::vector<f32> spd_y_values;
};

struct SdTreeNodeInfo {
    SdTreeNodeInfo()
        : pdf_x_values(LAMBDA_SAMPLES, 0.F), pdf_y_values(LAMBDA_SAMPLES, 0.F) {}

    SdTreeNodeInfo(const std::string &name, std::vector<f32> &&pdf_x_values,
                   std::vector<f32> &&pdf_y_values, std::vector<f32> &&samples)
        : name(name), pdf_x_values(std::move(pdf_x_values)),
          pdf_y_values(std::move(pdf_y_values)), samples(std::move(samples)) {}

    std::string name;
    std::vector<f32> pdf_x_values;
    std::vector<f32> pdf_y_values;

    std::vector<f32> samples;
};

struct SdTreeGuiInfo {
    // std::optional<SDTree> sd_tree_copy;
    std::optional<u32> selected_node;
    std::vector<SdTreeNodeInfo> nodes;
};

struct GuiState {
    ViewportSettings viewport_settings;
    Viewport viewport;
    RenderProgress render_progress;
    SceneInspector scene_inspector;
    SdTreeGuiInfo sd_tree;

    i32 dimension = 0;
    i32 num_points = 128;
    std::vector<f32> halton_x;
    std::vector<f32> halton_y;
};

#endif // GUISTATE_H
