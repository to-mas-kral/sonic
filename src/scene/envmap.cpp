#include "envmap.h"

#include <cmath>

#include "../math/aabb.h"
#include "../math/vecmath.h"
#include "sphere_square_mapping.h"

/*
 * Some of this code was taken / adapted from PBRTv4:
 * https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources#InfiniteAreaLight::Sample_Li
 * */
Envmap
Envmap::from_image(const ImageTexture &tex, const f32 scale,
                   const SquareMatrix4 &world_from_light) {
    std::vector<f32> sampling_grid(tex.width() * tex.height(), 0.f);

    const auto lambdas = SampledLambdas::new_sample_uniform(0.4f);

    spectral sum_rad = spectral::ZERO();

    for (int x = 0; x < tex.width(); ++x) {
        for (int y = 0; y < tex.height(); ++y) {
            const auto rgb = tex.fetch_rgb_texel(uvec2(x, y));
            const auto spec_illum = RgbSpectrumIlluminant(rgb, ColorSpace::sRGB);
            const auto spec = Spectrum(spec_illum);
            const auto rad = spec.eval(lambdas);

            sampling_grid[x + tex.width() * y] = (rgb.x + rgb.y + rgb.z) / 3.f;
            sum_rad += rad;
        }
    }

    // TODO: power calculation is wrong... this is a hack
    auto power = 4.f * M_PIf * scale * sum_rad.average() /
                 static_cast<f32>(tex.width() * tex.height());

    auto sampling_dist =
        PiecewiseDist2D::from_grid(sampling_grid, tex.width(), tex.height());

    return Envmap(tex, scale, power, world_from_light, std::move(sampling_dist));
}

spectral
Envmap::get_ray_radiance(const Ray &ray, const SampledLambdas &lambdas) const {
    const auto transformed_dir = light_from_world.transform_vec(ray.dir()).normalized();
    const auto xy = sphere_to_square(transformed_dir);

    const auto rgb = tex.fetch_rgb_texel_xycoords(xy);
    const auto spec_illum = RgbSpectrumIlluminant(rgb, ColorSpace::sRGB);

    return spec_illum.eval(lambdas) * scale;
}

std::optional<ShapeLightSample>
Envmap::sample(const point3 &illum_pos, const vec2 &xi, const SampledLambdas &lambdas,
               norm_vec3 *o_world_dir, vec2 *out_xy) const {
    auto [xy, pdf] = sampling_dist.sample(xi);

    if (pdf == 0.f) {
        return {};
    }

    if (out_xy) {
        *out_xy = xy;
    }

    const auto sphere_vec = square_to_sphere(xy);
    const auto world_dir = world_from_light.transform_vec(sphere_vec).normalized();
    if (o_world_dir != nullptr) {
        *o_world_dir = world_dir;
    }

    // From PBRT: change of variables factor from going from unit square to unit sphere
    pdf = pdf / (4.f * M_PIf);

    const auto rgb = tex.fetch_rgb_texel_xycoords(xy);
    const auto spec = RgbSpectrumIlluminant(rgb, ColorSpace::sRGB);

    const auto hit_pos = illum_pos + world_dir * (2.f * radius);

    return ShapeLightSample{
        .pos = hit_pos,
        .normal = (illum_pos - hit_pos).normalized(),
        .pdf = pdf,
        .emission = spec.eval(lambdas) * scale,
    };
}

f32
Envmap::pdf(const norm_vec3 &dir) const {
    const auto xy = sphere_to_square(light_from_world.transform_vec(dir).normalized());
    const auto pdf = sampling_dist.pdf(xy);

    return pdf / (4.f * M_PIf);
}

f32
Envmap::power() const {
    return m_power;
}

void
Envmap::set_bounds(const AABB &bounds) {
    const auto [c, r] = bounds.bounding_sphere();
    center = c;
    radius = r;
}
