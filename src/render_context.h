#ifndef PT_RENDER_CONTEXT_H
#define PT_RENDER_CONTEXT_H

#include "mesh.h"

/// Render Context is a collection of data needed for the render kernels to to their job.
class RenderContext {
public:
    explicit RenderContext(SharedVector<Mesh> &&meshes, u32 num_samples, u32 image_x, u32 image_y)
        : meshes(std::move(meshes)), num_samples(num_samples), image_x(image_x), image_y(image_y) {}

    __host__ __device__ u32 get_num_samples() const { return num_samples; }
    __host__ __device__ u32 get_image_x() const { return image_x; }
    __host__ __device__ u32 get_image_y() const { return image_y; }

private:
    SharedVector<Mesh> meshes{};
    // SharedVector<Material> materials;

    u32 num_samples = 16;
    u32 image_x = 800;

private:
    u32 image_y = 600;
};

#endif // PT_RENDER_CONTEXT_H
