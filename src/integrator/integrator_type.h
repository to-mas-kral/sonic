#ifndef PT_INTEGRATOR_TYPE_H
#define PT_INTEGRATOR_TYPE_H

#include "../utils/basic_types.h"

/// for underlying type size, see: https://github.com/CLIUtils/CLI11/issues/1086
enum class IntegratorType : u32 {
    Naive,
    MISNEE,
    PathGuiding
};

#endif // PT_INTEGRATOR_TYPE_H
