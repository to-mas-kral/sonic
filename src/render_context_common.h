#ifndef PT_RENDER_CONTEXT_COMMON_H
#define PT_RENDER_CONTEXT_COMMON_H

#include <cuda/std/atomic>
#include <cuda/std/optional>

#include "camera.h"
#include "envmap.h"
#include "framebuffer.h"
#include "geometry/bvh.h"
#include "geometry/geometry.h"
#include "geometry/intersection.h"
#include "geometry/ray.h"
#include "light.h"
#include "material.h"
#include "utils/numtypes.h"
#include "utils/shared_vector.h"

struct SceneAttribs {
    u32 resx{};
    u32 resy{};
    f32 fov{};
    mat4 camera_to_world = mat4(1.f);
};

/// Render Context is a collection of data needed for the render kernels to to their
/// job.
class RenderContext {
public:
    explicit RenderContext(u32 num_samples, SceneAttribs &attribs);

    __host__ u32 add_material(Material &&material);
    __host__ u32 add_light(Light &&light);
    __host__ u32 add_texture(Texture &&texture);
    __host__ void set_envmap(Envmap &&a_envmap) {
        envmap = std::move(a_envmap);
        has_envmap = true;
    };

    __host__ void make_acceleration_structure();
    __device__ cuda::std::optional<Intersection> intersect_scene(Ray &ray);

    // cuda::atomic<u32> ray_counter{0};

    Geometry geometry{};
    BVH bvh;

    SharedVector<Texture> textures = SharedVector<Texture>();
    SharedVector<Material> materials = SharedVector<Material>();
    SharedVector<Light> lights = SharedVector<Light>();

    Envmap envmap;
    bool has_envmap = false;
    Camera cam;

    Framebuffer fb;
    SceneAttribs attribs;

    u32 image_x;
    u32 image_y;
    u32 num_samples;

    const u32 THREADS_DIM_SIZE = 8;
    const dim3 THREADS_DIM = dim3(THREADS_DIM_SIZE, THREADS_DIM_SIZE);
    dim3 blocks_dim;
};

#endif // PT_RENDER_CONTEXT_COMMON_H
