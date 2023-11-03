#include "optix_pt.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>

#include "../geometry/intersection.h"
#include "../integrator/utils.h"
#include "../utils/numtypes.h"
#include "raygen.h"

extern "C" {
__constant__ PtParams params;
}

// TODO: could pack hit/miss into the prim_index...

// Payload structure
// 0 - hit or miss
// 1 - prim_index
// 2 - mesh_index
// 3 - barycentric x
// 4 - barycentric y
// 5, 6 - pos pointer
// 7, 8 - indices pointer

static __forceinline__ __device__ void set_payload_miss() { optixSetPayload_0(0); }

static __forceinline__ __device__ void set_payload_hit(u32 prim_index, u32 mesh_id,
                                                       float2 barycentrics,
                                                       CUdeviceptr pos,
                                                       CUdeviceptr indices) {
    optixSetPayload_0(1);
    optixSetPayload_1(prim_index);
    optixSetPayload_2(mesh_id);

    optixSetPayload_3(__float_as_uint(barycentrics.x));
    optixSetPayload_4(__float_as_uint(barycentrics.y));

    optixSetPayload_5(static_cast<u32>(pos));
    optixSetPayload_6((static_cast<u64>(pos) & 0xFFFF'FFFF'0000'0000U) >> 32U);
    optixSetPayload_7(static_cast<u32>(indices));
    optixSetPayload_8((static_cast<u64>(indices) & 0xFFFF'FFFF'0000'0000U) >> 32U);
}

extern "C" __global__ void __raygen__rg() {
    const uint3 pixel = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    auto rc = params.rc;
    auto pixel_index = params.fb->pixel_index(pixel.x, pixel.y);
    curandState *rand_state = &params.fb->get_rand_state()[pixel_index];

    auto ray = gen_ray(pixel.x, pixel.y, params.fb, rc);

    u32 depth = 1;
    vec3 throughput = vec3(1.f);
    vec3 radiance = vec3(0.f);

    while (true) {
        float3 raydir = make_float3(ray.dir.x, ray.dir.y, ray.dir.z);
        float3 rayorig = make_float3(ray.o.x, ray.o.y, ray.o.z);

        u32 did_hit = 0xdeadbeef, prim_index = 0xdeadbeef, mesh_id = 0xdeadbeef,
            bar_y = 0xdeadbeef, bar_z = 0xdeadbeef, pos_lo = 0xdeadbeef,
            pos_hi = 0xdeadbeef, indices_lo = 0xdeadbeef, indices_hi = 0xdeadbeef;
        optixTrace(params.gas_handle, rayorig, raydir, 0.0f, 1e16f, 0.0f,
                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0,
                   did_hit, prim_index, mesh_id, bar_y, bar_z, pos_lo, pos_hi, indices_lo,
                   indices_hi);

        CUdeviceptr d_pos = (static_cast<u64>(pos_hi) << 32U) | static_cast<u64>(pos_lo);
        CUdeviceptr d_indices =
            (static_cast<u64>(indices_hi) << 32U) | static_cast<u64>(indices_lo);

        if (did_hit) {
            f32 bar_y_f = __uint_as_float(bar_y);
            f32 bar_z_f = __uint_as_float(bar_z);
            vec3 bar = vec3(1.f - bar_y_f - bar_z_f, bar_y_f, bar_z_f);

            auto mesh = &params.meshes[mesh_id];
            auto material = &params.materials[mesh->material_id];

            vec3 *positions = (vec3 *)d_pos;
            u32 *indices = (u32 *)d_indices;

            u32 i0 = indices[3 * prim_index];
            u32 i1 = indices[3 * prim_index + 1];
            u32 i2 = indices[3 * prim_index + 2];

            vec3 p0 = positions[i0];
            vec3 p1 = positions[i1];
            vec3 p2 = positions[i2];

            vec3 pos = bar.x * p0 + bar.y * p1 + bar.z * p2;

            // TODO: adjust when mesh normals are added
            vec3 v0 = p1 - p0;
            vec3 v1 = p2 - p0;
            vec3 normal = glm::normalize(cross(v0, v1));
            if (glm::any(glm::isnan(normal))) {
                // Degenerate triangle...
                // TODO: HACK
                normal = glm::normalize(-ray.dir);
            }

            vec3 emission = vec3(0.f);
            if (mesh->has_light()) {
                emission = params.lights[mesh->light_id].emission();
            }

            if (glm::dot(-ray.dir, normal) < 0.f) {
                normal = -normal;
                emission = vec3(0.f);
            }

            Intersection its{
                .pos = pos,
                .normal = normal,
                .t = -1.f, // TODO: t used for anything ?
                .mesh = mesh,
            };

            vec3 sample_dir = material->sample(normal, -ray.dir, rand_state);
            // TODO: what to do when cos_theta is 0 ? this minimum value is a band-aid
            // solution...
            f32 cos_theta = max(glm::dot(normal, sample_dir), 0.0001f);

            f32 pdf = material->pdf(cos_theta);
            vec3 brdf = material->eval();

            radiance += throughput * emission;
            throughput *= brdf * cos_theta * (1.f / pdf);

            auto [should_terminate, roulette_compensation] =
                russian_roulette(depth, rand_state, throughput);

            if (should_terminate) {
                break;
            }

            throughput *= 1.f / roulette_compensation;

            Ray new_ray = spawn_ray(its, sample_dir);
            ray = new_ray;
            depth++;
        } else {
            // Ray has escaped the scene
            if (!rc->has_envmap) {
                radiance = vec3(0.f);
                break;
            } else {
                const Envmap *envmap = rc->get_envmap();
                vec3 envrad = envmap->sample(ray);
                radiance += throughput * envrad;
                break;
            }
        }
    }

    params.fb->get_pixels()[pixel_index] += radiance;
}

extern "C" __global__ void __miss__ms() { set_payload_miss(); }

extern "C" __global__ void __closesthit__ch() {
    PtHitGroupData *hit_data =
        reinterpret_cast<PtHitGroupData *>(optixGetSbtDataPointer());
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const u32 prim_index = optixGetPrimitiveIndex();

    set_payload_hit(prim_index, hit_data->mesh_id, barycentrics, hit_data->pos,
                    hit_data->indices);
}
