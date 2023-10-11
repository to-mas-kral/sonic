#ifndef PT_RENDER_CONTEXT_H
#define PT_RENDER_CONTEXT_H

#include "camera.h"
#include "framebuffer.h"
#include "light.h"
#include "material.h"
#include "shapes/mesh.h"
#include "shapes/triangle.h"
#include "utils/numtypes.h"
#include "utils/shared_vector.h"

/// Render Context is a collection of data needed for the render kernels to to their
/// job.
class RenderContext {
public:
    explicit RenderContext(u32 num_samples, u32 image_x, u32 image_y);

    __host__ void add_mesh(Mesh &&mesh);

    __host__ void add_material(Material &&material);

    __host__ void add_light(Light &&light);

    // TODO: use optional when available
    __device__ bool traverse_scene(Intersection &its, Ray &ray);

    __host__ __device__ const SharedVector<Material> &get_materials() const {
        return materials;
    }
    __host__ __device__ const SharedVector<Light> &get_lights() const { return lights; }
    __host__ __device__ const Camera &get_cam() const { return cam; }
    __host__ __device__ Framebuffer &get_framebuffer() { return fb; }
    __host__ __device__ u32 get_num_samples() const { return num_samples; }
    __host__ __device__ dim3 get_threads_dim() const { return THREADS_DIM; }
    __host__ __device__ dim3 get_blocks_dim() const { return blocks_dim; }

private:
    SharedVector<Mesh> meshes;
    SharedVector<Triangle> triangles; // It's a triangle vector, but could be a general
                                      // shape in the future...
    SharedVector<Material> materials;
    SharedVector<Light> lights;

    Camera cam;
    Framebuffer fb;

    u32 image_x;
    u32 image_y;
    u32 num_samples;

    const u32 THREADS_DIM_SIZE = 8;
    const dim3 THREADS_DIM = dim3(THREADS_DIM_SIZE, THREADS_DIM_SIZE);
    dim3 blocks_dim;
};

#endif // PT_RENDER_CONTEXT_H
