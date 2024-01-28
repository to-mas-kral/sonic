#ifndef PT_SAMPLED_SPECTRUM_H
#define PT_SAMPLED_SPECTRUM_H

#include "../math/math_utils.h"
#include "../math/vecmath.h"
#include "../utils/algs.h"
#include "../utils/basic_types.h"
#include "../utils/um_vector.h"
#include "cie_spectrums.h"

__device__ constexpr u32 N_SPECTRUM_SAMPLES = 4;
__device__ constexpr f32 PDF =
    1.f / (static_cast<f32>(LAMBDA_MAX) - static_cast<f32>(LAMBDA_MIN));

struct SampledSpectrum {
    SampledSpectrum() = default;

    __host__
        __device__ explicit SampledSpectrum(const CArray<f32, N_SPECTRUM_SAMPLES> &p_vals)
        : vals(p_vals) {}

    __host__ __device__ static SampledSpectrum
    make_constant(f32 constant) {
        SampledSpectrum sq{};
        sq.vals.fill(constant);
        return sq;
    }

    __host__ __device__ f32
    average() {
        f32 sum = 0.f;
        for (auto v : vals) {
            sum += v;
        }

        return sum / static_cast<f32>(N_SPECTRUM_SAMPLES);
    }

    __host__ __device__ f32
    max_component() const {
        f32 max = cuda::std::numeric_limits<f32>::min();
        for (f32 v : vals) {
            if (v > max) {
                max = v;
            }
        }

        return max;
    }

    __host__ __device__ void
    div_pdf(f32 pdf) {
        for (f32 &v : vals) {
            if (pdf != 0.f) {
                v /= pdf;
            }
        }
    }

    __host__ __device__ static SampledSpectrum
    ONE() {
        SampledSpectrum sq{};
        sq.vals.fill(1.f);
        return sq;
    }

    __host__ __device__ static SampledSpectrum
    ZERO() {
        SampledSpectrum sq{};
        sq.vals.fill(0.f);
        return sq;
    }

    __host__ __device__ SampledSpectrum
    operator+(const SampledSpectrum &other) const {
        SampledSpectrum sq{};
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            sq.vals[i] = vals[i] + other.vals[i];
        }

        return sq;
    }

    __host__ __device__ SampledSpectrum &
    operator+=(const SampledSpectrum &other) {
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            vals[i] += other.vals[i];
        }

        return *this;
    }

    __host__ __device__ SampledSpectrum
    operator-(const SampledSpectrum &other) const {
        SampledSpectrum sq{};
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            sq.vals[i] = vals[i] - other.vals[i];
        }

        return sq;
    }

    __host__ __device__ SampledSpectrum &
    operator-=(const SampledSpectrum &other) {
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            vals[i] -= other.vals[i];
        }

        return *this;
    }

    __host__ __device__ SampledSpectrum
    operator*(const SampledSpectrum &other) const {
        SampledSpectrum sq{};
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            sq.vals[i] = vals[i] * other.vals[i];
        }

        return sq;
    }

    __host__ __device__ SampledSpectrum &
    operator*=(const SampledSpectrum &other) {
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            vals[i] *= other.vals[i];
        }

        return *this;
    }

    __host__ __device__ SampledSpectrum
    operator*(f32 val) const {
        SampledSpectrum sq{};
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            sq.vals[i] = vals[i] * val;
        }

        return sq;
    }

    __host__ __device__ SampledSpectrum &
    operator*=(f32 val) {
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            vals[i] *= val;
        }

        return *this;
    }

    __host__ __device__ SampledSpectrum
    operator/(f32 div) const {
        SampledSpectrum sq{};
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            sq.vals[i] = vals[i] / div;
        }

        return sq;
    }

    __host__ __device__ SampledSpectrum
    operator/(const SampledSpectrum &other) const {
        SampledSpectrum sq{};
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            sq.vals[i] = vals[i] / other.vals[i];
        }

        return sq;
    }

    __host__ __device__ f32 &
    operator[](u32 index) {
        return vals[index];
    }

    CArray<f32, N_SPECTRUM_SAMPLES> vals;
};

struct SampledLambdas {
    __host__ __device__ static SampledLambdas
    new_sample_uniform(f32 rand) {
        SampledLambdas sl{};

        f32 lambda_min = static_cast<f32>(LAMBDA_MIN);
        f32 lambda_max = static_cast<f32>(LAMBDA_MAX);

        // Sample first wavelength
        sl.lambdas[0] = annie::lerp(rand, lambda_min, lambda_max);

        if constexpr (N_SPECTRUM_SAMPLES > 1) {
            // Initialize remaining wavelenghts
            f32 delta = (lambda_max - lambda_min) / static_cast<f32>(N_SPECTRUM_SAMPLES);

            for (int i = 1; i < N_SPECTRUM_SAMPLES; i++) {
                sl.lambdas[i] = sl.lambdas[i - 1] + delta;
                if (sl.lambdas[i] > lambda_max) {
                    sl.lambdas[i] = lambda_min + (sl.lambdas[i] - lambda_max);
                }
            }
        }

        return sl;
    }

    __host__ __device__ static SampledLambdas
    new_mock() {
        SampledLambdas sl{};
        sl.lambdas.fill(400.f);
        return sl;
    }

    __host__ __device__ vec3
    to_xyz(const SampledSpectrum &radiance);

    __host__ __device__ const f32 &
    operator[](u32 index) const {
        return lambdas[index];
    }

    CArray<f32, N_SPECTRUM_SAMPLES> lambdas;
};

using spectral = SampledSpectrum;

#endif // PT_SAMPLED_SPECTRUM_H
