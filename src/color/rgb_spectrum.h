#ifndef PT_RGB_SPECTRUM_H
#define PT_RGB_SPECTRUM_H

#include "../math/vecmath.h"
#include "color_space.h"
#include "rgb2spec.h"
#include "spectrum.h"

/// Bounded reflectance spectrum [0; 1]
struct RgbSpectrum {
    static RgbSpectrum
    make(const tuple3 &rgb);

    __host__ __device__ static RgbSpectrum
    from_coeff(const tuple3 &sigmoig_coeff) {
        return RgbSpectrum{
            .sigmoid_coeff = sigmoig_coeff,
        };
    }

    static RgbSpectrum
    make_empty() {
        return RgbSpectrum{
            .sigmoid_coeff = tuple3(0.f),
        };
    }

    __host__ __device__ f32
    eval_single(f32 lambda) const {
        return RGB2Spec::eval(sigmoid_coeff, lambda);
    }

    __host__ __device__ spectral
    eval(const SampledLambdas &lambdas) const {
        spectral sq{};
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            sq[i] = eval_single(lambdas[i]);
        }

        return sq;
    }

    tuple3 sigmoid_coeff = tuple3(0.f);
};

struct RgbSpectrumUnbounded : public RgbSpectrum {
    static RgbSpectrumUnbounded
    make(const tuple3 &rgb);

    __host__ __device__ f32
    eval_single(f32 lambda) const {
        return scale * RGB2Spec::eval(sigmoid_coeff, lambda);
    }

    __host__ __device__ spectral
    eval(const SampledLambdas &lambdas) const {
        spectral sq{};
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            sq[i] = eval_single(lambdas[i]);
        }

        return sq;
    }

    f32 scale = 1.f;
};

struct RgbSpectrumIlluminant : public RgbSpectrumUnbounded {
    static RgbSpectrumIlluminant
    make(const tuple3 &rgb, ColorSpace color_space);

    __host__ __device__ f32
    eval_single(f32 lambda) const {
        f32 res = scale * RGB2Spec::eval(sigmoid_coeff, lambda);
        const DenseSpectrum *illuminant = nullptr;
        switch (color_space) {
        case ColorSpace::sRGB:
            illuminant = &CIE_65;
            break;
        default:
            assert(false);
        }

        /// TODO: this is a hack for normalizing the D65 illuminant to having a luminance
        /// of 1
        res *= illuminant->eval_single(lambda) * (CIE_Y_INTEGRAL / 10789.7637f);

        return res;
    }

    __host__ __device__ spectral
    eval(const SampledLambdas &lambdas) const {
        spectral sq{};
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            sq[i] = eval_single(lambdas[i]);
        }

        return sq;
    }

    // TODO: best to store this out-of-band in the case if illuminant textures...
    ColorSpace color_space = ColorSpace::sRGB;
};

#endif // PT_RGB_SPECTRUM_H
