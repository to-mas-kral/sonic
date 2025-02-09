#ifndef PT_ENVMAP_H
#define PT_ENVMAP_H

#include "../geometry/geometry_container.h"
#include "../geometry/ray.h"
#include "../math/piecewise_dist.h"
#include "../spectrum/spectral_quantity.h"
#include "texture.h"

class AABB;

class Envmap {
public:
    static Envmap
    from_image(const ImageTexture &tex, f32 scale, const SquareMatrix4 &world_from_light);

    spectral
    get_ray_radiance(const Ray &ray, const SampledLambdas &lambdas) const;

    std::optional<ShapeLightSample>
    sample(const point3 &illum_pos, const vec2 &xi, const SampledLambdas &lambdas,
           norm_vec3 *o_world_dir = nullptr, vec2 *out_xy = nullptr) const;

    f32
    pdf(const norm_vec3 &dir) const;

    f32
    power() const;

    u32
    light_id() const {
        return m_light_id;
    }

    void
    set_light_id(const u32 light_id) {
        m_light_id = light_id;
    }

    void
    set_bounds(const AABB &bounds);

private:
    explicit Envmap(const ImageTexture &tex, const f32 scale, const f32 power,
                    const SquareMatrix4 &world_from_light,
                    PiecewiseDist2D &&sampling_dist)
        : tex{tex}, scale{scale}, m_power{power}, world_from_light{world_from_light},
          light_from_world{world_from_light.inverse()},
          sampling_dist(std::move(sampling_dist)) {}

    ImageTexture tex;
    f32 scale{1.F};
    f32 m_power{0.F};
    SquareMatrix4 world_from_light;
    SquareMatrix4 light_from_world;
    PiecewiseDist2D sampling_dist;

    vec3 center{0.F};
    f32 radius{1.F};
    u32 m_light_id{0};
};

#endif // PT_ENVMAP_H
