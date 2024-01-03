#ifndef PT_ENVMAP_H
#define PT_ENVMAP_H

#include <cmath>
#include <string>

#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <tinyexr.h>

#include "color/spectrum.h"
#include "color/sampled_spectrum.h"
#include "geometry/ray.h"
#include "math/vecmath.h"
#include "texture.h"
#include "utils/basic_types.h"
#include "utils/cuda_err.h"
#include "utils/um_vector.h"

__global__ static void
calc_texture(int width, int height, f32 *img, cudaTextureObject_t tex_obj) {
    for (int v = 0; v < height; v++) {
        f32 vp = (f32)v / (f32)height;
        f32 sin_theta = sin(M_PIf * f32(v + 0.5f) / f32(height));
        for (int u = 0; u < width; ++u) {
            f32 up = (f32)u / (f32)width;
            auto elem = tex2D<float4>(tex_obj, up, vp);

            // FIXME: this should calculate the power of the RGBSpectrum...
            img[u + v * width] = avg<f32>(elem.x, elem.y, elem.z);
            img[u + v * width] *= sin_theta;
        }
    }
}
/*
 * Some of this code was taken / adapted from PBRTv4:
 * https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources#InfiniteAreaLight::Sample_Li
 * */

class Envmap : Texture {
public:
    Envmap() : Texture(){};

    explicit Envmap(const std::string &texture_path, const mat4 &to_world_transform)
        : Texture(Texture::make(texture_path, true)),
          to_world_transform(to_world_transform.inverse()) {
        UmVector<f32> img(width * height);
        img.assume_all_init();

        calc_texture<<<1, 1>>>(width, height, img.get_ptr(), tex_obj);
        CUDA_CHECK(cudaDeviceSynchronize())
        sampling_dist = PiecewiseDist2D(img, width, height);
    };

    __device__ __forceinline__ spectral
    get_ray_radiance(const Ray &ray, const SampledLambdas &lambdas) const {
        // TODO: correct coordinates for environment mapping...
        /*Ray tray = Ray(ray);
        tray.dir = tray.dir.normalize();
        tray.transform(to_world_transform);*/

        // Mapping from ray direction to UV on equirectangular texture
        // (1 / 2pi, 1 / pi)
        const vec2 pi_reciprocals = vec2(0.1591f, 0.3183f);
        vec2 uv = vec2(atan2(-ray.dir.z, -ray.dir.x), asin(ray.dir.y));
        uv *= pi_reciprocals;
        uv += 0.5;

        auto coeff = fetch(uv);
        return RgbSpectrum::from_coeff(coeff).eval(lambdas);
    };

    /// Returns radiance, direction and pdf
    __device__ __forceinline__ CTuple<tuple3, norm_vec3, f32>
    sample(const vec2 &sample) const {
        auto [uv, pdf] = sampling_dist.sample(sample);
        if (pdf == 0.f) {
            return {tuple3(0.f), norm_vec3(0.f), pdf};
        }

        f32 theta = uv[1] * M_PIf;
        f32 phi = uv[0] * 2.f * M_PIf;
        f32 sin_theta = sin(theta);
        f32 cos_theta = cos(theta);
        f32 sin_phi = sin(phi);
        f32 cos_phi = cos(phi);
        norm_vec3 dir =
            vec3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi).normalized();

        pdf /= (2.f * sqr(M_PIf) * sin_theta);
        if (sin_theta == 0.f) {
            pdf = 0.f;
        }

        return {fetch(uv), dir, pdf};
    };

    __device__ __forceinline__ f32
    pdf(const vec3 &dir) const {
        const vec2 pi_reciprocals = vec2(0.1591f, 0.3183f);
        vec2 uv = vec2(atan2(-dir.z, -dir.x), asin(dir.y));
        uv *= pi_reciprocals;
        uv += 0.5;
        uv.y = -uv.y;

        f32 theta = uv[1] * M_PIf;
        f32 sin_theta = sin(theta);

        return sampling_dist.pdf(uv) / (2.f * sqr(M_PIf) * sin_theta);
    };

private:
    mat4 to_world_transform = mat4::identity();
    PiecewiseDist2D sampling_dist{};
};

#endif // PT_ENVMAP_H
