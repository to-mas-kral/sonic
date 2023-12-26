#ifndef PT_SPECTRUM_H
#define PT_SPECTRUM_H

#include "../math/math_utils.h"
#include "../math/vecmath.h"
#include "../utils/basic_types.h"
#include "../utils/um_vector.h"
#include "cie_spectrums.h"

__device__ constexpr u32 N_SPECTRUM_SAMPLES = 4;
__device__ constexpr f32 PDF =
    1.f / (static_cast<f32>(LAMBDA_MAX) - static_cast<f32>(LAMBDA_MIN));

struct SpectralQuantity {
    SpectralQuantity() = default;

    __host__ __device__ explicit SpectralQuantity(
        const CArray<f32, N_SPECTRUM_SAMPLES> &p_vals)
        : vals(p_vals) {}

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

    __host__ __device__ static SpectralQuantity
    ONE() {
        SpectralQuantity sq{};
        sq.vals.fill(1.f);
        return sq;
    }

    __host__ __device__ static SpectralQuantity
    ZERO() {
        SpectralQuantity sq{};
        sq.vals.fill(0.f);
        return sq;
    }

    __host__ __device__ SpectralQuantity
    operator+(const SpectralQuantity &other) const {
        SpectralQuantity sq{};
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            sq.vals[i] = vals[i] + other.vals[i];
        }

        return sq;
    }

    __host__ __device__ SpectralQuantity &
    operator+=(const SpectralQuantity &other) {
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            vals[i] += other.vals[i];
        }

        return *this;
    }

    __host__ __device__ SpectralQuantity
    operator*(const SpectralQuantity &other) const {
        SpectralQuantity sq{};
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            sq.vals[i] = vals[i] * other.vals[i];
        }

        return sq;
    }

    __host__ __device__ SpectralQuantity &
    operator*=(const SpectralQuantity &other) {
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            vals[i] *= other.vals[i];
        }

        return *this;
    }

    __host__ __device__ SpectralQuantity
    operator*(f32 val) const {
        SpectralQuantity sq{};
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            sq.vals[i] = vals[i] * val;
        }

        return sq;
    }

    __host__ __device__ SpectralQuantity &
    operator*=(f32 val) {
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            vals[i] *= val;
        }

        return *this;
    }

    __host__ __device__ SpectralQuantity
    operator/(f32 div) const {
        SpectralQuantity sq{};
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            sq.vals[i] = vals[i] / div;
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

    __host__ __device__ vec3
    to_xyz(const SpectralQuantity &radiance);

    __host__ __device__ const f32 &
    operator[](u32 index) const {
        return lambdas[index];
    }

    CArray<f32, N_SPECTRUM_SAMPLES> lambdas;
};

class DenseSpectrum {
public:
    __host__ __device__ constexpr static DenseSpectrum
    from_static(const CArray<f32, LAMBDA_RANGE> &data) {
        DenseSpectrum ds{};
        ds.vals = data.data();
        ds.is_static = true;

        return ds;
    }

    __host__ static DenseSpectrum
    make(const UmVector<f32> &&data) {
        DenseSpectrum ds{};

        throw std::runtime_error(
            "Loading dense spectra from dynamically loaded data is not implemented yet");

        return ds;
    }

    // TODO: should be linearly interpolated or not ?
    __host__ __device__ f32
    eval_single(f32 lambda) const {
        assert(lambda >= LAMBDA_MIN && lambda <= LAMBDA_MAX);
        u32 index = lround(lambda) - LAMBDA_MIN;
        return vals[index];
    }

    __host__ __device__ inline SpectralQuantity
    eval(const SampledLambdas &sl) const {
        SpectralQuantity sq{};
        for (int i = 0; i < N_SPECTRUM_SAMPLES; i++) {
            sq[i] = eval_single(sl.lambdas[i]);
        }

        return sq;
    }

private:
    const f32 *vals;
    bool is_static;
};

using spectral = SpectralQuantity;

__device__ const DenseSpectrum CIE_X = DenseSpectrum::from_static(CIE_X_RAW);
__device__ const DenseSpectrum CIE_Y = DenseSpectrum::from_static(CIE_Y_RAW);
__device__ const DenseSpectrum CIE_Z = DenseSpectrum::from_static(CIE_Z_RAW);

__host__ __device__ inline vec3
SampledLambdas::to_xyz(const SpectralQuantity &radiance) {
    SpectralQuantity x =
        CIE_X.eval(static_cast<const SampledLambdas &>(*this)) * radiance;
    SpectralQuantity y =
        CIE_Y.eval(static_cast<const SampledLambdas &>(*this)) * radiance;
    SpectralQuantity z =
        CIE_Z.eval(static_cast<const SampledLambdas &>(*this)) * radiance;

    x.div_pdf(PDF);
    y.div_pdf(PDF);
    z.div_pdf(PDF);

    f32 x_xyz = x.average() / CIE_Y_INTEGRAL;
    f32 y_xyz = y.average() / CIE_Y_INTEGRAL;
    f32 z_xyz = z.average() / CIE_Y_INTEGRAL;
    return vec3(x_xyz, y_xyz, z_xyz);
}

#endif // PT_SPECTRUM_H
