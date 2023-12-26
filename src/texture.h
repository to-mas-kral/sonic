#ifndef PT_TEXTURE_H
#define PT_TEXTURE_H

#include <cmath>
#include <string>

#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <tinyexr.h>

#include "geometry/ray.h"
#include "math/piecewise_dist.h"
#include "math/vecmath.h"
#include "utils/basic_types.h"
#include "utils/cuda_box.h"
#include "utils/cuda_err.h"
#include "utils/um_vector.h"

class Texture {
public:
    Texture() = default;

    Texture(cudaTextureObject_t tex_obj, cudaArray_t texture_storage_array, i32 width,
            i32 height)
        : tex_obj(tex_obj), texture_storage_array(texture_storage_array), width(width),
          height(height) {}

    static Texture
    make(const std::string &texture_path, bool is_rgb);

    __device__ __forceinline__ tuple3
    fetch(const vec2 &uv) const {
        auto ret = tex2D<float4>(tex_obj, uv.x, 1.f - uv.y);
        return tuple3(ret.x, ret.y, ret.z);
    };

    // Do I even need these ?
    Texture(Texture &&other) noexcept;
    Texture &
    operator=(Texture &&other) noexcept;

    ~Texture();

protected:
    cudaTextureObject_t tex_obj = 0;
    cudaArray_t texture_storage_array = nullptr;
    i32 width = 0;
    i32 height = 0;
};

#endif // PT_TEXTURE_H
