
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

__host__ void RenderContext::add_mesh(MeshParams mp) {
    u32 num_indices = mp.indices->len();
    u32 num_vertices = mp.pos->len();

    u32 indices_index = indices.len();
    for (int i = 0; i < mp.indices->len(); i++) {
        indices.push(std::move((*mp.indices)[i]));
    }

    u32 pos_index = pos.len();
    for (int i = 0; i < mp.pos->len(); i++) {
        pos.push(std::move((*mp.pos)[i]));
    }

    cuda::std::optional<u32> normals_index = cuda::std::nullopt;
    if (mp.normals != nullptr) {
        normals_index = {normals.len()};
        assert(mp.normals->len() == mp.pos->len());

        for (int i = 0; i < mp.normals->len(); i++) {
            normals.push(std::move((*mp.normals)[i]));
        }
    }

    cuda::std::optional<u32> uvs_index = cuda::std::nullopt;
    if (mp.uvs != nullptr) {
        uvs_index = {uvs.len()};
        assert(mp.uvs->len() == mp.pos->len());

        for (int i = 0; i < mp.uvs->len(); i++) {
            uvs.push(std::move((*mp.uvs)[i]));
        }
    }

    auto mesh_id = meshes.len();
    auto mesh = Mesh(indices_index, pos_index, mp.material_id, mp.light_id, this,
                     num_indices, num_vertices, normals_index, uvs_index);
    meshes.push(std::move(mesh));

    for (int i = 0; i < mp.indices->len(); i += 3) {
        Triangle triangle = Triangle(i / 3, mesh_id);
        triangles.push(std::move(triangle));
    }
}

__host__ void RenderContext::fixup_geometry_pointers() {
    // Fixup the triangle-mesh pointers
    for (int i = 0; i < triangles.len(); i++) {
        u32 mesh_id = triangles[i].mesh_id;
        triangles[i].set_mesh(&meshes[mesh_id]);
    }
}

__host__ void RenderContext::make_acceleration_structure() {
    const int MAX_PRIMS_IN_NODE = 8;
    bvh = BVH(&triangles, MAX_PRIMS_IN_NODE);
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
