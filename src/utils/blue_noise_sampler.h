#ifndef PT_BLUE_NOISE_SAMPLER_H
#define PT_BLUE_NOISE_SAMPLER_H

#include <cuda/std/array>

#include "blue_noise_tiles.h"
#include "numtypes.h"

class BlueNoiseSampler {
public:
    __device__ __forceinline__ f32 sample(i32 pixel_i, i32 pixel_j, i32 sampleIndex,
                                          i32 sampleDimension) {
        // wrap arguments
        pixel_i = pixel_i & 127;
        pixel_j = pixel_j & 127;
        sampleIndex = sampleIndex & 255;
        sampleDimension = sampleDimension & 255;

        // xor index based on optimized ranking
        i32 rankedSampleIndex =
            sampleIndex ^ rankingTile[sampleDimension + (pixel_i + pixel_j * 128) * 8];

        // fetch value in sequence
        i32 value = sobol_256spp_256d[sampleDimension + rankedSampleIndex * 256];

        // If the dimension is optimized, xor sequence value based on optimized scrambling
        value =
            value ^ scramblingTile[(sampleDimension % 8) + (pixel_i + pixel_j * 128) * 8];

        // convert to float and return
        f32 v = (0.5f + value) / 256.0f;
        return v;
    }

    template <size_t N>
    __device__ __forceinline__ cuda::std::array<f32, N>
    create_samples(i32 pixel_i, i32 pixel_j, i32 sample_index) {
        cuda::std::array<f32, N> samples;
        for (int i = 0; i < N; i++) {
            samples[i] = sample(pixel_i, pixel_j, sample_index, dim++);
        }

        return samples;
    }

private:
    i32 dim = 0;

}; // namespace BlueNoiseSampler

#endif // PT_BLUE_NOISE_SAMPLER_H
