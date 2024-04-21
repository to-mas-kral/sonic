#ifndef PT_SPECTRUM_H
#define PT_SPECTRUM_H

#include "../utils/chunk_allocator.h"
#include "cie_spectrums.h"
#include "color_space.h"
#include "sampled_spectrum.h"
#include "spectrum_consts.h"

class DenseSpectrum {
public:
    constexpr static DenseSpectrum
    make(const Array<f32, LAMBDA_RANGE> &data) {
        DenseSpectrum ds{};
        ds.vals = data.data();

        return ds;
    }

    f32
    eval_single(f32 lambda) const;

    SampledSpectrum
    eval(const SampledLambdas &sl) const;

private:
    const f32 *vals;
};

constexpr DenseSpectrum CIE_X = DenseSpectrum::make(CIE_X_RAW);
constexpr DenseSpectrum CIE_Y = DenseSpectrum::make(CIE_Y_RAW);
constexpr DenseSpectrum CIE_Z = DenseSpectrum::make(CIE_Z_RAW);
constexpr DenseSpectrum CIE_65 = DenseSpectrum::make(CIE_D65_RAW);

class PiecewiseSpectrum {
public:
    static PiecewiseSpectrum
    make(const Span<f32> &data);

    f32
    eval_single(f32 lambda) const;

    inline SampledSpectrum
    eval(const SampledLambdas &sl) const;

private:
    const f32 *vals;
    u32 size_half;
};

class ConstantSpectrum {
public:
    static constexpr ConstantSpectrum
    make(f32 val) {
        ConstantSpectrum cs{};
        cs.val = val;
        return cs;
    }

    f32
    eval_single(f32 lambda) const;

    inline SampledSpectrum
    eval(const SampledLambdas &sl) const;

private:
    f32 val;
};

/// Bounded reflectance spectrum [0; 1]
struct RgbSpectrum {
    static RgbSpectrum
    make(const tuple3 &rgb);

    static RgbSpectrum
    from_coeff(const tuple3 &sigmoig_coeff);

    static RgbSpectrum
    make_empty();

    f32
    eval_single(f32 lambda) const;

    spectral
    eval(const SampledLambdas &lambdas) const;

    tuple3 sigmoid_coeff = tuple3(0.f);
};

struct RgbSpectrumUnbounded : RgbSpectrum {
    static RgbSpectrumUnbounded
    make(const tuple3 &rgb);

    f32
    eval_single(f32 lambda) const;

    spectral
    eval(const SampledLambdas &lambdas) const;

    f32 scale = 1.f;
};

struct RgbSpectrumIlluminant : RgbSpectrumUnbounded {
    static RgbSpectrumIlluminant
    make(const tuple3 &rgb, ColorSpace color_space);

    f32
    eval_single(f32 lambda) const;

    spectral
    eval(const SampledLambdas &lambdas) const;

    // TODO: best to store this out-of-band in the case if illuminant textures...
    ColorSpace color_space = ColorSpace::sRGB;
};

enum class SpectrumType {
    Constant,
    Dense,
    PiecewiseLinear,
    Rgb,
    RgbUnbounded,
};

struct Spectrum {
    explicit
    Spectrum(DenseSpectrum ds)
        : type{SpectrumType::Dense}, dense_spectrum{ds} {}

    explicit
    Spectrum(PiecewiseSpectrum ps, ChunkAllocator<> *spectrum_allocator)
        : type{SpectrumType::PiecewiseLinear} {
        piecewise_spectrum = spectrum_allocator->allocate<PiecewiseSpectrum>();
        *piecewise_spectrum = ps;
    }

    explicit
    Spectrum(ConstantSpectrum cs)
        : type{SpectrumType::Constant}, constant_spectrum{cs} {}

    explicit
    Spectrum(RgbSpectrum rs, ChunkAllocator<> *spectrum_allocator)
        : type{SpectrumType::Rgb} {
        rgb_spectrum = spectrum_allocator->allocate<RgbSpectrum>();
        *rgb_spectrum = rs;
    }

    explicit
    Spectrum(RgbSpectrumUnbounded rs, ChunkAllocator<> *spectrum_allocator)
        : type{SpectrumType::RgbUnbounded} {
        rgb_spectrum_unbounded = spectrum_allocator->allocate<RgbSpectrumUnbounded>();
        *rgb_spectrum_unbounded = rs;
    }

    SampledSpectrum
    eval(const SampledLambdas &lambdas) const;

    f32
    eval_single(f32 lambda) const;

    SpectrumType type;
    union {
        DenseSpectrum dense_spectrum;
        PiecewiseSpectrum *piecewise_spectrum;
        ConstantSpectrum constant_spectrum{};
        RgbSpectrum *rgb_spectrum;
        RgbSpectrumUnbounded *rgb_spectrum_unbounded;
    };
};

#endif // PT_SPECTRUM_H
