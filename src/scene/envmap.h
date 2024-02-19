#ifndef PT_ENVMAP_H
#define PT_ENVMAP_H

#include "../color/sampled_spectrum.h"
#include "../geometry/ray.h"
#include "../math/vecmath.h"
#include "texture.h"

class Envmap : ImageTexture {
public:
    Envmap() : ImageTexture(){};

    explicit Envmap(const std::string &texture_path, const mat4 &to_world_transform);

    spectral
    get_ray_radiance(const Ray &ray, const SampledLambdas &lambdas) const;

    /// Returns radiance, direction and pdf
    Tuple<tuple3, norm_vec3, f32>
    sample(const vec2 &sample);

    f32
    pdf(const vec3 &dir);

private:
    mat4 to_world_transform = mat4::identity();
    PiecewiseDist2D sampling_dist{};
};

#endif // PT_ENVMAP_H
