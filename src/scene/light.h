#ifndef PT_LIGHT_H
#define PT_LIGHT_H

#include "../geometry/geometry.h"
#include "../integrator/intersection.h"
#include "../utils/basic_types.h"
#include "envmap.h"

class Light;

struct LightSample {
    point3 pos{};
    norm_vec3 normal{1.f, 0.f, 0.f};
    f32 pdf{};
    Light const *light{nullptr};
    spectral emission;
};

enum class LightType : u8 {
    ShapeLight,
    EnvmapLight,
};

class ShapeLight {
public:
    ShapeLight(const ShapeIndex shape, const Emitter &emitter)
        : shape{shape}, emitter{emitter} {}

    ShapeLightSample
    sample(const Intersection &its, const vec3 &shape_rng, const SampledLambdas &lambdas,
           const Geometry &geom) const {
        auto sample = geom.sample_shape(shape, its.pos, shape_rng);

        const norm_vec3 pl = (sample.pos - its.pos).normalized();
        const f32 cos_light = vec3::dot(sample.normal, -pl);
        const f32 pl_mag_sq = (sample.pos - its.pos).length_squared();
        // Probability of sampling this point in the measure of solid angle from the
        // illuminated point...
        const f32 pdf_light = sample.pdf * (pl_mag_sq / cos_light);
        sample.pdf = pdf_light;

        sample.emission = emission(lambdas);
        return sample;
    }

    // TODO: twosided lights in general
    f32
    power(const Geometry &geom) const {
        auto area = geom.shape_area(shape);
        if (emitter.twosided) {
            area *= 2.f;
        }
        return M_PI * emitter.power() * area;
    }

    f32
    area(const Geometry &geom) const {
        return geom.shape_area(shape);
    }

    spectral
    emission(const SampledLambdas &lambdas) const {
        return emitter.emission(lambdas);
    }

private:
    ShapeIndex shape;
    Emitter emitter;
};

class EnvmapLight {
public:
    explicit
    EnvmapLight(Envmap *envmap)
        : envmap{envmap} {}

    std::optional<ShapeLightSample>
    sample(const point3 &pos, const vec3 &shape_rng,
           const SampledLambdas &lambdas) const {
        return envmap->sample(pos, vec2(shape_rng.x, shape_rng.y), lambdas);
    }

    f32
    power() const {
        return envmap->power();
    }

private:
    Envmap *envmap{nullptr};
};

class Light {
public:
    explicit
    Light(const ShapeLight &shape_light)
        : light_type{LightType::ShapeLight}, inner({.shape_light = shape_light}) {}

    explicit
    Light(const EnvmapLight &envmap_light)
        : light_type{LightType::EnvmapLight}, inner({.envmap_light = envmap_light}) {}

    /// PDF is returned w.r.t solid angle !
    std::optional<LightSample>
    sample(const f32 light_sampler_pdf, const vec3 &shape_rng,
           const SampledLambdas &lambdas, const Intersection &its,
           const Geometry &geom) const {
        ShapeLightSample sample{};

        switch (light_type) {
        case LightType::ShapeLight:
            sample = inner.shape_light.sample(its, shape_rng, lambdas, geom);
            break;
        case LightType::EnvmapLight: {
            const auto s_opt = inner.envmap_light.sample(its.pos, shape_rng, lambdas);
            if (!s_opt.has_value()) {
                return {};
            }

            sample = s_opt.value();
            break;
        }
        default:
            panic();
        }

        return LightSample{
            .pos = sample.pos,
            .normal = sample.normal,
            .pdf = sample.pdf * light_sampler_pdf,
            .light = this,
            .emission = sample.emission,
        };
    }

    f32
    power(const Geometry &geom) const {
        switch (light_type) {
        case LightType::ShapeLight:
            return inner.shape_light.power(geom);
        case LightType::EnvmapLight:
            return inner.envmap_light.power();
        default:
            panic();
        }
    }

    spectral
    emission(const SampledLambdas &lambdas) const {
        switch (light_type) {
        case LightType::ShapeLight:
            return inner.shape_light.emission(lambdas);
        case LightType::EnvmapLight:
            panic();
        default:
            panic();
        }
    }

    f32
    area(const Geometry &geom) const {
        switch (light_type) {
        case LightType::ShapeLight:
            return inner.shape_light.area(geom);
        case LightType::EnvmapLight:
            panic();
        default:
            panic();
        }
    }

private:
    LightType light_type;
    union {
        ShapeLight shape_light;
        EnvmapLight envmap_light;
    } inner;
};

#endif // PT_LIGHT_H
