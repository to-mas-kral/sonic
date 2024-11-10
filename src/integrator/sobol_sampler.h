#ifndef SOBOL_SAMPLER_H
#define SOBOL_SAMPLER_H

#include "../math/vecmath.h"
#include "../utils/basic_types.h"
#include "sobol_matrices.h"

#include <algorithm>
#include <cmath>

class SobolSampler {
public:
    void
    init_frame(const uvec2 &pixel, const uvec2 &resolution, const u32 p_frame,
               const u32 spp) {
        index = p_frame + spp * ((pixel.y * resolution.x) + pixel.x);
        dimension = 0;
    }

    f32
    sample() {
        return sobol_sample(index, dimension++);
    }

    vec2
    sample2() {
        const auto x = sample();
        const auto y = sample();
        return vec2(x, y);
    }

    vec3
    sample3() {
        const auto x = sample();
        const auto y = sample();
        const auto z = sample();
        return vec3(x, y, z);
    }

    vec2
    sample_camera() {
        return sample2();
    }
    
    /// Brent Burley, Practical Hash-based Owen Scrambling, Journal of Computer Graphics
    /// Techniques (JCGT), vol. 9, no. 4, 1-20, 2020
    /// Available online http : jcgt.org/published/0009/04/01/
    static u32
    laine_karras_permutation(u32 x) {
        x += seed;
        x ^= x * 0x6c50b47cU;
        x ^= x * 0xb82f1e52U;
        x ^= x * 0xc7afe638U;
        x ^= x * 0x8d22f6e6U;
        return x;
    }

    static u32
    pbrt_permutation(u32 v) {
        v ^= v * 0x3d20adea;
        v += seed;
        v *= (seed >> 16) | 1;
        v ^= v * 0x05526c56;
        v ^= v * 0x53a22864;
        return v;
    }

    static u32
    reverse_bits(u32 x) {
#if defined __clang__ && __has_builtin(__builtin_bitreverse32)
        return __builtin_bitreverse32(x);
#else
        /// https://www.topcoder.com/thrive/articles/A%20bit%20of%20fun:%20fun%20with%20bits
        x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
        x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
        x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
        x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
        x = ((x & 0xffff0000) >> 16) | ((x & 0x0000ffff) << 16);
        return x;
#endif
    }

    static u32
    scramble(u32 x) {
        x = reverse_bits(x);
        // x = laine_karras_permutation(x);
        x = pbrt_permutation(x);
        return reverse_bits(x);
    }

    /// This was taken from PBRTv-4, but Cycles has almost the same code...
    static f32
    sobol_sample(u64 a, i32 p_dimension) {
        if (p_dimension >= N_SOBOL_DIMENSIONS) {
            p_dimension = 0;
        }

        u32 v = 0;
        for (int i = p_dimension * SOBOL_MATRIX_SIZE; a != 0; a >>= 1, i++) {
            if (a & 1) {
                v ^= SOBOL_MATRICES[i];
            }
        }

        v = scramble(v);

        return std::min(v * 0x1p-32f, ONE_MINUS_EPS);
    }

    i32 dimension{0};
    u64 index{0};
    static constexpr u32 seed = 1297302534;
};

#endif // SOBOL_SAMPLER_H
