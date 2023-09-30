#ifndef PT_RENDER_CONTEXT_H
#define PT_RENDER_CONTEXT_H

#include "framebuffer.h"
#include "mesh.h"
#include "utils/numtypes.h"

/// Render Context is a collection of data needed for the render kernels to to their
/// job.
class RenderContext {
public:
    explicit RenderContext(SharedVector<Mesh> &&meshes, u32 num_samples, u32 image_x,
                           u32 image_y)
        : meshes(std::move(meshes)), num_samples(num_samples) {

        blocks = (num_samples + threads_per_block - 1U) / threads_per_block;
        fb = Framebuffer(image_x, image_y);
    };

    __host__ __device__ u32 get_num_samples() const { return num_samples; }

    __host__ __device__ u32 get_threads_per_block() const { return threads_per_block; }
    __host__ __device__ u32 get_blocks() const { return blocks; }

    __host__ __device__ Framebuffer &get_fb() { return fb; }

private:
    SharedVector<Mesh> meshes;
    // SharedVector<Material> materials;

    Framebuffer fb;

    u32 num_samples;

    u32 threads_per_block = 256;
    u32 blocks;
};

#endif // PT_RENDER_CONTEXT_H
