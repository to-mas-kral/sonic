#ifndef PT_TEXTURE_H
#define PT_TEXTURE_H

#include <cmath>
#include <string>

#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <tinyexr.h>

#include "geometry/ray.h"
#include "utils/cuda_err.h"
#include "utils/numtypes.h"
#include "utils/shared_vector.h"

class Texture {
public:
    Texture() : width(0), height(0) {}

    explicit Texture(const std::string &texture_path);

    __device__ __forceinline__ vec3 sample(vec2 uv) const {
        auto ret = tex2D<float4>(tex_obj, uv.x, 1.f - uv.y);
        return vec3(ret.x, ret.y, ret.z);
    };

    // Do I even need these ?
    Texture(Texture &&other) noexcept;
    Texture &operator=(Texture &&other) noexcept;

    ~Texture();

protected:
    cudaTextureObject_t tex_obj = 0;
    cudaArray_t texture_storage_array = nullptr;
    i32 width = 0;
    i32 height = 0;
};

#endif // PT_TEXTURE_H
