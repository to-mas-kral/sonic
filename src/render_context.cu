
#include "render_context_common.h"

#include "geometry/intersection.h"
#include "geometry/ray.h"

RenderContext::RenderContext(u32 num_samples, SceneAttribs &attribs)
    : num_samples(num_samples), image_x(attribs.resx), image_y(attribs.resy),
      attribs(attribs) {

    u32 blocks_x = (attribs.resx + THREADS_DIM_SIZE - 1U) / THREADS_DIM_SIZE;
    u32 blocks_y = (attribs.resy + THREADS_DIM_SIZE - 1U) / THREADS_DIM_SIZE;
    blocks_dim = dim3(blocks_x, blocks_y);

    f32 aspect = static_cast<f32>(attribs.resx) / static_cast<f32>(attribs.resy);
    cam = Camera(attribs.fov, aspect);
    fb = Framebuffer(attribs.resx, attribs.resy, blocks_dim, THREADS_DIM);
}

/*__device__ bool RenderContext::intersect_scene(Intersection &its, Ray &ray) {
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
}*/

__device__ cuda::std::optional<Intersection> RenderContext::intersect_scene(Ray &ray) {
    return bvh.intersect(ray, cuda::std::numeric_limits<f32>::max());
}

__host__ void RenderContext::make_acceleration_structure() {
    const int MAX_PRIMS_IN_NODE = 8;
    bvh = BVH(&geometry.meshes.triangles, MAX_PRIMS_IN_NODE);
}

__host__ u32 RenderContext::add_material(Material &&material) {
    u32 mat_id = materials.len();
    materials.push(std::move(material));
    return mat_id;
}

__host__ u32 RenderContext::add_light(Light &&light) {
    u32 light_id = lights.len();
    lights.push(std::move(light));
    return light_id;
}

__host__ u32 RenderContext::add_texture(Texture &&texture) {
    u32 texture_id = textures.len();
    textures.push(std::move(texture));
    return texture_id;
}
