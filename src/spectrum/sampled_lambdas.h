#ifndef SAMPLEDLAMBDAS_H
#define SAMPLEDLAMBDAS_H

#include "spectral_quantity.h"

class SampledLambdas {
public:
    static SampledLambdas
    sample_uniform(f32 xi);

    static SampledLambdas
    sample_visual_importance(f32 xi);

    static f32
    pdf_visual_importance(f32 lambda);

    static SampledLambdas
    new_mock();

    SampledLambdas(const std::array<f32, N_SPECTRUM_SAMPLES> &lambdas,
                   const spectral &pdfs)
        : pdfs(pdfs), m_lambdas(lambdas) {}

    SampledLambdas() {}

    vec3
    to_xyz(const SpectralQuantity &radiance) const;

    void
    terminate_secondary();

    bool
    is_secondary_terminated() const {
        return m_is_secondary_terminated;
    }

    const f32 &
    operator[](u32 index) const;

    spectral pdfs{};

private:
    bool m_is_secondary_terminated{false};
    std::array<f32, N_SPECTRUM_SAMPLES> m_lambdas{};
};

#endif // SAMPLEDLAMBDAS_H
