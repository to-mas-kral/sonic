#ifndef PT_SCENE_H
#define PT_SCENE_H

#include "emitter.h"
#include "envmap.h"
#include "geometry/geometry.h"
#include "integrator/light_sampler.h"
#include "material.h"
#include "scene/light.h"
#include "texture.h"

struct Scene {
    __host__ void set_envmap(Envmap &&a_envmap) {
        envmap = std::move(a_envmap);
        has_envmap = true;
    };

    __host__ void add_mesh(MeshParams mp);
    __host__ void add_sphere(SphereParams sp);

    __host__ u32 add_material(Material &&material);
    __host__ u32 add_texture(Texture &&texture);

    __host__ void init_light_sampler();
    
    __device__ __forceinline__ cuda::std::optional<LightSample>
    sample_lights(f32 sample) const {
        return light_sampler.sample(lights, sample);
    }

    Geometry geometry{};
    LightSampler light_sampler{};
    SharedVector<Light> lights{};

    SharedVector<Texture> textures = SharedVector<Texture>();
    SharedVector<Material> materials = SharedVector<Material>();

    Envmap envmap{};
    bool has_envmap = false;
};

#endif // PT_SCENE_H
