
#include "render_context.h"

#include "geometry/intersection.h"
#include "geometry/ray.h"

RenderContext::RenderContext(u32 num_samples, u32 image_x, u32 image_y)
    : meshes(SharedVector<Mesh>(8)), triangles(SharedVector<Triangle>(64)),
      num_samples(num_samples), image_x(image_x), image_y(image_y) {

    u32 blocks_x = (image_x + THREADS_DIM_SIZE - 1U) / THREADS_DIM_SIZE;
    u32 blocks_y = (image_y + THREADS_DIM_SIZE - 1U) / THREADS_DIM_SIZE;
    blocks_dim = dim3(blocks_x, blocks_y);

    f32 aspect = static_cast<f32>(image_x) / static_cast<f32>(image_y);
    cam = Camera(19.5, aspect);
    fb = Framebuffer(image_x, image_y, blocks_dim, THREADS_DIM);
}

__device__ bool RenderContext::traverse_scene(Intersection &its, Ray &ray) {
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

__host__ void RenderContext::add_mesh(Mesh &&mesh) {
    meshes.push(std::move(mesh));
    auto &indices = meshes.last().get_indices();

    for (int i = 0; i < indices.len(); i += 3) {
        Triangle triangle = Triangle(&meshes.last(), i);
        triangles.push(std::move(triangle));
    }
}

__host__ void RenderContext::add_material(Material &&material) {
    materials.push(std::move(material));
}

__host__ void RenderContext::add_light(Light &&light) { lights.push(std::move(light)); }
