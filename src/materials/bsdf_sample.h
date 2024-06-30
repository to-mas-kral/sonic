#ifndef PT_BSDF_SAMPLE_H
#define PT_BSDF_SAMPLE_H

#include "../color/sampled_spectrum.h"
#include "../integrator/shading_frame.h"
#include "../utils/basic_types.h"

struct BSDFSample {
    spectral bsdf{};
    f32 pdf{};
    bool did_refract{false};
    ShadingFrame sframe;
};
#endif // PT_BSDF_SAMPLE_H
