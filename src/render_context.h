#ifndef PT_RENDER_CONTEXT_H
#define PT_RENDER_CONTEXT_H

#include "utils/cuda_err.h"

#include "framebuffer.h"
#include "shapes/mesh.h"
#include "shapes/triangle.h"
#include "utils/numtypes.h"

/// Render Context is a collection of data needed for the render kernels to to their
/// job.
class RenderContext {
public:
    explicit RenderContext(u32 num_samples, u32 image_x, u32 image_y)
        : meshes(SharedVector<Mesh>(8)), triangles(SharedVector<Triangle>(64)),
          num_samples(num_samples), image_x(image_x), image_y(image_y) {

        blocks = (num_samples + threads_per_block - 1U) / threads_per_block;

        sample_accum = SharedVector(vec3(0.f, 0.f, 0.f), num_samples);
    };

    __host__ void add_mesh(Mesh &&mesh) {
        meshes.push(std::move(mesh));
        auto &indices = meshes.last().get_indices();

        for (int i = 0; i < indices.len(); i += 3) {
            Triangle triangle = Triangle(&meshes.last(), i);
            triangles.push(std::move(triangle));
        }
    }

    __host__ __device__ u64 pixel_index(u64 x, u64 y) const { return (y * image_x) + x; }

    __host__ __device__ SharedVector<vec3> &get_sample_accum() { return sample_accum; }
    __host__ __device__ u32 get_num_samples() const { return num_samples; }
    __host__ __device__ u32 get_threads_per_block() const { return threads_per_block; }
    __host__ __device__ u32 get_blocks() const { return blocks; }

private:
    SharedVector<Mesh> meshes;
    SharedVector<Triangle> triangles; // It's a triangle vector, but could be a general
                                      // shape in the future...

    // SharedVector<Material> materials;

    // An accumulation buffer for different samples
    SharedVector<vec3> sample_accum;
    u32 image_x;
    u32 image_y;
    u32 num_samples;
    u32 threads_per_block = 256;
    u32 blocks;
};

#endif // PT_RENDER_CONTEXT_H
