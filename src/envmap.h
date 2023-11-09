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

    explicit Envmap(const std::string &texture_path, mat4 to_world_transform)
        : Texture(texture_path), to_world_transform(glm::inverse(to_world_transform)){};

    __device__ __forceinline__ vec3 sample(Ray &ray) const {
        // FIXME: correct coordinates for environment mapping...
        /*Ray tray = Ray(ray);
        tray.dir = glm::normalize(tray.dir);
        tray.transform(to_world_transform);*/

        // Mapping from ray direction to UV on equirectangular texture
        // (1 / 2pi, 1 / pi)
        const vec2 pi_reciprocals = vec2(0.1591f, 0.3183f);
        vec2 uv = vec2(atan2(-ray.dir.z, -ray.dir.x), asin(ray.dir.y));
        uv *= pi_reciprocals;
        uv += 0.5;

        auto ret = tex2D<float4>(tex_obj, uv.x, 1.f - uv.y);
        return vec3(ret.x, ret.y, ret.z);
    };

private:
    mat4 to_world_transform = mat4(1.);
};

#endif // PT_ENVMAP_H
