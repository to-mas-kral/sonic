#include "optix_pt.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>

#include "../color/sampled_spectrum.h"
#include "../math/vecmath.h"
#include "../utils/basic_types.h"
#include "../utils/sampler.h"
#include "bdpt_nee_integrator.h"
#include "hit_type.h"
#include "mis_nee_integrator.h"
#include "raygen.h"
#include "utils.h"

extern "C" {
__constant__ PtParams params;
}

static __forceinline__ __device__ void
set_payload_miss() {
    optixSetPayload_0(static_cast<u32>(HitType::Miss));
}

static __forceinline__ __device__ void
set_payload_hit_triangle(u32 prim_index, u32 mesh_id, float2 barycentrics) {
    optixSetPayload_0(static_cast<u32>(HitType::Triangle));
    optixSetPayload_1(prim_index);
    optixSetPayload_2(mesh_id);

    optixSetPayload_3(__float_as_uint(barycentrics.x));
    optixSetPayload_4(__float_as_uint(barycentrics.y));
}

static __forceinline__ __device__ void
set_payload_hit_sphere(u32 prim_index, f32 t) {
    optixSetPayload_0(static_cast<u32>(HitType::Sphere));
    optixSetPayload_1(prim_index);
    optixSetPayload_2(__float_as_uint(t));
}

extern "C" __global__ void
__raygen__rg() {
    const uint3 pixel = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    auto rc = params.rc;
    auto pixel_index = ((dim.y - 1U - pixel.y) * dim.x) + pixel.x;

    Sampler sampler{};
    sampler.init_frame(uvec2(pixel.x, pixel.y), uvec2(dim.x, dim.y), params.frame);

    auto cam_sample = sampler.sample2();
    auto ray =
        gen_ray(pixel.x, pixel.y, dim.x, dim.y, cam_sample, rc->cam, params.cam_to_world);

    SampledLambdas lambdas = SampledLambdas::new_sample_uniform(sampler.sample());

    spectral radiance = spectral::ZERO();

    if (params.integrator_type == IntegratorType::Naive ||
        params.integrator_type == IntegratorType::MISNEE) {
        radiance = integrator_mis_nee(params, ray, sampler, lambdas);
    } else if (params.integrator_type == IntegratorType::BDPTNEE) {
        radiance = integrator_bdpt_nee(params, ray, sampler, lambdas);
    }

    params.fb->get_pixels()[pixel_index] += lambdas.to_xyz(radiance);
}

extern "C" __global__ void
__miss__ms() {
    set_payload_miss();
}

/*
 * From Nvidia docs:
 * "It is generally more efficient to have one hit shader handle multiple primitive
 * types (by switching on the value of optixGetPrimitiveType), rather than have several
 * hit shaders that implement the same ray behavior but differ only in the type of
 * geometry they expect."
 * */
extern "C" __global__ void
__closesthit__ch() {
    auto *hit_data = reinterpret_cast<PtHitGroupData *>(optixGetSbtDataPointer());

    auto type = optixGetPrimitiveType();

    if (type == OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
        // Triangles
        const float2 barycentrics = optixGetTriangleBarycentrics();
        const u32 prim_index = optixGetPrimitiveIndex();

        set_payload_hit_triangle(prim_index, hit_data->mesh.mesh_id, barycentrics);
    } else {
        // Spheres
        // CHECK: maybe add sphere_id to hit_data... would be safer if multiple
        //  sphere GASes are used in the future...
        const u32 sphere_index = optixGetSbtGASIndex();
        f32 t = optixGetRayTmax();

        set_payload_hit_sphere(sphere_index, t);
    }
}
