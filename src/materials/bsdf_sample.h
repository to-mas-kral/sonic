#ifndef PT_BSDF_SAMPLE_H
#define PT_BSDF_SAMPLE_H

#include "../color/sampled_spectrum.h"
#include "../integrator/shading_frame.h"
#include "../utils/basic_types.h"

struct BSDFSample {
    BSDFSample(const spectral &bsdf, const f32 pdf, const bool did_refract,
               const ShadingFrame &sframe)
        : bsdf(bsdf), pdf(pdf), did_refract(did_refract), sframe(sframe) {}

    spectral bsdf;
    f32 pdf;
    bool did_refract;
    ShadingFrame sframe;
};
#endif // PT_BSDF_SAMPLE_H
