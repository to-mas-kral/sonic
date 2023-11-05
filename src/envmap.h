#ifndef PT_ENVMAP_H
#define PT_ENVMAP_H

#include <cmath>
#include <string>

#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <tinyexr.h>

#include "geometry/ray.h"
#include "texture.h"
#include "utils/cuda_err.h"
#include "utils/numtypes.h"
#include "utils/shared_vector.h"

class Envmap : Texture {
public:
    Envmap() : Texture(){};

    explicit Envmap(const std::string &texture_path) : Texture(texture_path){};

    __device__ __forceinline__ vec3 sample(Ray &ray) const {
        // Mapping from ray direction to UV on equirectangular texture
        // (1 / 2pi, 1 / pi)
        const vec2 pi_reciprocals = vec2(0.1591f, 0.3183f);
        vec2 uv = vec2(atan2(ray.dir.z, ray.dir.x), asin(ray.dir.y));
        uv *= pi_reciprocals;
        uv += 0.5;

        auto ret = tex2D<float4>(tex_obj, uv.x, 1.f - uv.y);
        return vec3(ret.x, ret.y, ret.z);
    };
};

#endif // PT_ENVMAP_H
