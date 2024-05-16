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
Envmap(SpectrumTexture *tex, const f32 scale)
    : tex{tex}, scale{scale} {
    /*std::vector<f32> img(width * height, 0.f);

    sampling_dist = PiecewiseDist2D(img, width, height);*/
}

/// Code taken from PBRTv4.
/// Via source code from Clarberg: Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD.
vec2
sphere_to_square(const vec3 &arg_dir) {
    // Change coordinates from world-space to paper-space
    const auto dir = vec3(arg_dir.x, arg_dir.z, arg_dir.y);
    // assert(dir.is_normalized());
    assert(sqr(dir.length()) > 0.999 && sqr(dir.length()) < 1.001);
    const auto x = std::abs(dir.x);
    const auto y = std::abs(dir.y);
    const auto z = std::abs(dir.z);

    // Compute the radius r
    const auto r = safe_sqrt(1.f - z);
    // Compute the argument to atan (detect a=0 to avoid div-by-zero)
    const auto a = std::max(x, y);
    f32 b = std::min(x, y);
    if (a == 0.f) {
        b = 0.f;
    } else {
        b /= a;
    }

    // Polynomial approximation of atan(x)*2/pi, x=b
    // Coefficients for 6th degree minimax approximation of atan(x)*2/pi,
    // x=[0,1].
    constexpr auto T1 = 0.406758566246788489601959989e-5f;
    constexpr auto T2 = 0.636226545274016134946890922156f;
    constexpr auto T3 = 0.61572017898280213493197203466e-2f;
    constexpr auto T4 = -0.247333733281268944196501420480f;
    constexpr auto T5 = 0.881770664775316294736387951347e-1f;
    constexpr auto T6 = 0.419038818029165735901852432784e-1f;
    constexpr auto T7 = -0.251390972343483509333252996350e-1f;

    auto phi = T6 + T7 * b;
    phi = T5 + phi * b;
    phi = T4 + phi * b;
    phi = T3 + phi * b;
    phi = T2 + phi * b;
    phi = T1 + phi * b;

    // Extend phi if the input is in the range 45-90 degrees (u<v)
    if (x < y) {
        phi = 1.f - phi;
    }

    // Find (u,v) based on (r,phi)
    auto v = phi * r;
    auto u = r - v;

    if (dir.z < 0.f) {
        // southern hemisphere -> mirror u,v
        auto tmp = u;
        u = v;
        v = tmp;

        u = 1.f - u;
        v = 1.f - v;
    }

    // Move (u,v) to the correct quadrant based on the signs of (x,y)
    u = std::copysign(u, dir.x);
    v = std::copysign(v, dir.y);

    // Transform (u,v) from [-1,1] to [0,1]
    return vec2(0.5f * (u + 1.f), 0.5f * (v + 1.f));
}

spectral
Envmap::get_ray_radiance(const Ray &ray, const SampledLambdas &lambdas) const {
    const auto uv = sphere_to_square(ray.dir);
    return tex->fetch(uv).eval(lambdas) * scale;
}

/*Tuple<Spectrum, norm_vec3, f32>
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
}*/

/*f32
Envmap::pdf(const vec3 &dir) {
    const vec2 pi_reciprocals = vec2(0.1591f, 0.3183f);
    vec2 uv = vec2(std::atan2(-dir.z, -dir.x), std::asin(dir.y));
    uv *= pi_reciprocals;
    uv += 0.5;
    uv.y = -uv.y;

    f32 theta = uv[1] * M_PIf;
    f32 sin_theta = std::sin(theta);

    return sampling_dist.pdf(uv) / (2.f * sqr(M_PIf) * sin_theta);
}*/
