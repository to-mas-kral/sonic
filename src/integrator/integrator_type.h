#ifndef PT_INTEGRATOR_TYPE_H
#define PT_INTEGRATOR_TYPE_H

/// sefor underlying type size, see: https://github.com/CLIUtils/CLI11/issues/1086
enum class IntegratorType : u32 {
    Naive,
    MISNEE,
};

#endif // PT_INTEGRATOR_TYPE_H
