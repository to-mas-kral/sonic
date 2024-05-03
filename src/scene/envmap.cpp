#include "envmap.h"

#include <cmath>

/*
 * Some of this code was taken / adapted from PBRTv4:
 * https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources#InfiniteAreaLight::Sample_Li
 * */

/*__global__ static void
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
}*/

Envmap::
Envmap(const std::string &texture_path, const mat4 &to_world_transform)
    : ImageTexture(ImageTexture::make(texture_path, true)),
      to_world_transform(to_world_transform.inverse()) {
    std::vector<f32> img(width * height, 0.f);

    sampling_dist = PiecewiseDist2D(img, width, height);
}

spectral
Envmap::get_ray_radiance(const Ray &ray, const SampledLambdas &lambdas) const {
    // TODO: correct coordinates for environment mapping...
    /*Ray tray = Ray(ray);
    tray.dir = tray.dir.normalize();
    tray.transform(to_world_transform);*/

    // Mapping from ray direction to UV on equirectangular texture
    // (1 / 2pi, 1 / pi)
    const vec2 pi_reciprocals = vec2(0.1591f, 0.3183f);
    vec2 uv = vec2(std::atan2(-ray.dir.z, -ray.dir.x), std::asin(ray.dir.y));
    uv *= pi_reciprocals;
    uv += 0.5;

    return fetch_spectrum(uv).eval(lambdas);
}

Tuple<Spectrum, norm_vec3, f32>
Envmap::sample(const vec2 &sample) {
    auto [uv, pdf] = sampling_dist.sample(sample);
    if (pdf == 0.f) {
        return {Spectrum(ConstantSpectrum::make(0.f)), norm_vec3(0.f), pdf};
    }

    f32 theta = uv[1] * M_PIf;
    f32 phi = uv[0] * 2.f * M_PIf;
    f32 sin_theta = std::sin(theta);
    f32 cos_theta = std::cos(theta);
    f32 sin_phi = std::sin(phi);
    f32 cos_phi = std::cos(phi);
    norm_vec3 dir =
        vec3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi).normalized();

    pdf /= (2.f * sqr(M_PIf) * sin_theta);
    if (sin_theta == 0.f) {
        pdf = 0.f;
    }

    return {fetch_spectrum(uv), dir, pdf};
}

f32
Envmap::pdf(const vec3 &dir) {
    const vec2 pi_reciprocals = vec2(0.1591f, 0.3183f);
    vec2 uv = vec2(std::atan2(-dir.z, -dir.x), std::asin(dir.y));
    uv *= pi_reciprocals;
    uv += 0.5;
    uv.y = -uv.y;

    f32 theta = uv[1] * M_PIf;
    f32 sin_theta = std::sin(theta);

    return sampling_dist.pdf(uv) / (2.f * sqr(M_PIf) * sin_theta);
}
