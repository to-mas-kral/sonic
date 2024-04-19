#ifndef PT_UTILS_H
#define PT_UTILS_H

#include "../color/sampled_spectrum.h"
#include "../geometry/ray.h"
#include "../math/vecmath.h"
#include "../utils/basic_types.h"

#include <algorithm>

/// Randomly selects if a path should be terminated based on its throughput.
/// Roulette is only applied after the first 3 bounces.
/// Returns true if path should be terminated. If not, also returns roulette compensation.
inline Option<f32>
russian_roulette(u32 depth, f32 u, const spectral &throughput) {
    if (depth > 3) {
        f32 survival_prob = 1.f - std::max(throughput.max_component(), 0.05f);

        if (u < survival_prob) {
            return {};
        } else {
            f32 roulette_compensation = 1.f - survival_prob;
            return roulette_compensation;
        }
    } else {
        return 1.f;
    }
}

struct ShadingGeometry {
    ///  w_i is incident direction and w_o is outgoing direction.
    ///  w_o goes "towards the viewer" and w_i "towards the light"
    static ShadingGeometry
    make(const norm_vec3 &normal, const norm_vec3 &wi, const norm_vec3 &wo) {
        // TODO: what to do when cos_theta is 0 ? this minimum value is a band-aid
        /*The f() function performs the required coordinate frame conversion and then
         * queries the BxDF. The rare case in which the wo direction lies exactly in the
         * surface’s tangent plane often leads to not-a-number (NaN) values in BxDF
         * implementations that further propagate and may eventually contaminate the
         * rendered image. The BSDF avoids this case by immediately returning a
         * zero-valued SampledSpectrum. */
        f32 cos_theta = std::max(vec3::dot(normal, wi), 0.0001f);
        norm_vec3 h = (wi + wo).normalized();
        f32 noh = vec3::dot(normal, h);
        f32 nowo = vec3::dot(normal, wo);
        f32 nowi = vec3::dot(normal, wi);
        f32 howo = vec3::dot(h, wo);

        return ShadingGeometry{
            .cos_theta = cos_theta,
            .nowo = nowo,
            .nowi = nowi,
            .noh = noh,
            .howo = howo,
            .h = h,
        };
    }

    /// Detects the edge cases where wo or wi lie exactly on the tangent plane
    bool
    is_degenerate() const {
        return nowi == 0.f || nowo == 0.f;
    }

    f32 cos_theta;
    /// Dot product between normal and w_o.
    f32 nowo;
    /// Dot product between normal and w_i.
    f32 nowi;
    /// Dot product between normal and halfway vector.
    f32 noh;
    /// Dot product between halfway vector and w_o.
    f32 howo;
    /// Halfway vector
    norm_vec3 h;
};

/// Specific case where 1 sample is taken from each distribution.
inline f32
mis_power_heuristic(f32 fpdf, f32 gpdf) {
    return sqr(fpdf) / (sqr(fpdf) + sqr(gpdf));
}

#endif // PT_UTILS_H
