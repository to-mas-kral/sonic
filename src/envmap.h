#ifndef PT_ENVMAP_H
#define PT_ENVMAP_H

#include <cmath>
#include <string>

#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <tinyexr.h>

#include "geometry/ray.h"
#include "utils/numtypes.h"
#include "utils/shared_vector.h"

class Envmap {
public:
    Envmap() : pixels(SharedVector<vec3>()), width(0), height(0) {}

    Envmap(const std::string &texture_path) : pixels(SharedVector<vec3>()) {
        f32 *out = nullptr; // width * height * RGBA
        const char *err = nullptr;
        int ret = LoadEXR(&out, &width, &height, texture_path.c_str(), &err);

        if (ret != TINYEXR_SUCCESS) {
            if (err) {
                spdlog::error("EXR loading error: {}", err);
                FreeEXRErrorMessage(err);
            }
        }

        assert(width > 0 && height > 0);
        pixels = SharedVector<vec3>(width * height);
        pixels.assume_all_init();

        // Have to copy image data to GPU-accessible memory...
        // RGBA -> RGB
        for (size_t j = 0; j < size_t(width * height); j++) {
            vec3 rgb = vec3();
            rgb.r = out[4 * j + 0];
            rgb.g = out[4 * j + 1];
            rgb.b = out[4 * j + 2];
            pixels[j] = rgb;
        }
    }

    __device__ vec3 sample(Ray &ray) const {
        // Mapping from ray direction to UV on equirectangular texture
        // (1 / 2pi, 1 / pi)
        const vec2 pi_reciprocals = vec2(0.1591, 0.3183);
        vec2 uv = vec2(atan2(ray.dir.z, ray.dir.x), asin(ray.dir.y));
        uv *= pi_reciprocals;
        uv += 0.5;

        i32 x = static_cast<i32>(uv.x * (static_cast<f32>(width - 1)));
        i32 y = height - static_cast<i32>(uv.y * (static_cast<f32>(height - 1)));
        return pixels[y * width + x];
    };

private:
    SharedVector<vec3> pixels;
    i32 width = 0;
    i32 height = 0;
};

#endif // PT_ENVMAP_H
