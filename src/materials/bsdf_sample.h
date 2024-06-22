#ifndef PT_BSDF_SAMPLE_H
#define PT_BSDF_SAMPLE_H

#include "../color/sampled_spectrum.h"
#include "../math/vecmath.h"
#include "../utils/basic_types.h"

struct BSDFSample {
    spectral bsdf{};
    norm_vec3 wi{1.f, 0.f, 0.f};
    f32 pdf{};
    bool did_refract{false};
};
#endif // PT_BSDF_SAMPLE_H
