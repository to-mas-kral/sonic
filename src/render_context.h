#ifndef PT_RENDER_CONTEXT_H
#define PT_RENDER_CONTEXT_H

#include "camera.h"
#include "framebuffer.h"
#include "light.h"
#include "material.h"
#include "shapes/mesh.h"
#include "shapes/triangle.h"
#include "utils/cuda_err.h"
#include "utils/numtypes.h"
#include "utils/shared_vector.h"

/// Render Context is a collection of data needed for the render kernels to to their
/// job.
class RenderContext {
public:
    explicit RenderContext(u32 num_samples, u32 image_x, u32 image_y)
        : meshes(SharedVector<Mesh>(8)), triangles(SharedVector<Triangle>(64)),
          num_samples(num_samples), image_x(image_x), image_y(image_y) {

        u32 blocks_x = (image_x + THREADS_DIM_SIZE - 1U) / THREADS_DIM_SIZE;
        u32 blocks_y = (image_y + THREADS_DIM_SIZE - 1U) / THREADS_DIM_SIZE;
        blocks_dim = dim3(blocks_x, blocks_y);

        f32 aspect = static_cast<f32>(image_x) / static_cast<f32>(image_y);
        cam = Camera(19.5, aspect);
        fb = Framebuffer(image_x, image_y, blocks_dim, THREADS_DIM);
    };

    __host__ void add_mesh(Mesh &&mesh) {
        meshes.push(std::move(mesh));
        auto &indices = meshes.last().get_indices();

        for (int i = 0; i < indices.len(); i += 3) {
            Triangle triangle = Triangle(&meshes.last(), i);
            triangles.push(std::move(triangle));
        }
    }

    __host__ void add_material(Material &&material) {
        materials.push(std::move(material));
    }

    __host__ void add_light(Light &&light) { lights.push(std::move(light)); }

    // TODO: use optional when available
    __device__ bool traverse_scene(Intersection &its, Ray &ray) {
        // TODO: make some acceleration structure...
        bool its_found = false;
        f32 min_t = cuda::std::numeric_limits<f32>::max();

        for (int t = 0; t < triangles.len(); t++) {
            auto tri = &triangles[t];
            Intersection tri_its;
            if (tri->intersect(tri_its, ray)) {
                if (tri_its.t < min_t) {
                    min_t = tri_its.t;
                    its = tri_its;
                }
                its_found = true;
            }
        }

        return its_found;
    }

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
