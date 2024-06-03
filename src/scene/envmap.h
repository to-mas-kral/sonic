#ifndef PT_ENVMAP_H
#define PT_ENVMAP_H

#include "../color/sampled_spectrum.h"
#include "../geometry/geometry.h"
#include "../geometry/ray.h"
#include "../math/piecewise_dist.h"
#include "texture.h"

struct AABB;

class Envmap {
public:
    explicit
    Envmap(ImageTexture tex, f32 scale, const SquareMatrix4 &world_from_light);

    spectral
    get_ray_radiance(const Ray &ray, const SampledLambdas &lambdas) const;

    std::optional<ShapeLightSample>
    sample(const point3 &illum_pos, const vec2 &sample, const SampledLambdas &lambdas, vec3 *o_world_dir = nullptr) const;

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
    ImageTexture tex;
    f32 scale{1.};
    f32 m_power{0.f};
    SquareMatrix4 world_from_light{};
    SquareMatrix4 light_from_world{};
    PiecewiseDist2D sampling_dist{};

    vec3 center{0.f};
    f32 radius{1.f};
    u32 m_light_id{0};
};

#endif // PT_ENVMAP_H
