#ifndef SETTINGS_H
#define SETTINGS_H

#include "integrator/integrator_type.h"
#include "utils/basic_types.h"

struct Settings {
    bool silent = false;
    bool render_normals = false;
    u32 start_frame = 0;
    u32 spp = 32;
    IntegratorType integrator_type = IntegratorType::MISNEE;
};

#endif // SETTINGS_H
