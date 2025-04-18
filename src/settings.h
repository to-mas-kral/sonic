#ifndef SETTINGS_H
#define SETTINGS_H

#include "integrator/integrator_type.h"
#include "utils/basic_types.h"

struct Settings {
    bool silent = false;
    bool load_only = false;
    bool no_gui = false;
    bool save_progress = false;
    std::string scene_path{};
    std::string out_filename;
    u32 start_frame = 0;
    u32 spp = 32;
    u32 num_threads = 0;
    IntegratorType integrator_type = IntegratorType::MISNEE;
};

#endif // SETTINGS_H
