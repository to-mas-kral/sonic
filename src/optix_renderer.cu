#include <array>
#include <vector>

#include "optix_common.h"
#include "optix_renderer.h"

OptixRenderer::OptixRenderer(Scene *sc, OptixDeviceContext context, OptixAS *optixAS)
    : optixAS(optixAS) {
    auto pipeline_compile_options = make_optix_pipeline_compile_options(13, "params");

    OptixModuleCompileOptions module_compile_options = {};
    make_optix_module(context, &pipeline_compile_options, &module, "optix_pt.ptx",
                      &module_compile_options);

    OptixModule sphere_is_module{};
    OptixModuleCompileOptions sphere_module_compile_options = {};
    OptixBuiltinISOptions builtin_is_options = {};
    builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    OPTIX_CHECK(optixBuiltinISModuleGet(context, &sphere_module_compile_options,
                                        &pipeline_compile_options, &builtin_is_options,
                                        &sphere_is_module));

    OptixProgramGroupOptions pg_options = {};
    make_optix_program_groups(context, module, sphere_is_module, &pg_options, &raygen_pg,
                              &miss_pg, &hitgroup_pg, &sphere_hitgroup_pg);

    const u32 max_trace_depth = 1;
    std::array<OptixProgramGroup, 4> pgs = {raygen_pg, miss_pg, hitgroup_pg,
                                            sphere_hitgroup_pg};
    link_optix_pipeline(context, &pipeline_compile_options, max_trace_depth, pgs,
                        &pipeline);

    /*
     * Set up shader binding table
     * */
    {
        PtRayGenSbtRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pg, &rg_sbt));
        raygen_record.set(&rg_sbt);

        PtMissSbtRecord ms_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg, &ms_sbt));
        miss_record.set(&ms_sbt);

        auto pos = &sc->geometry.meshes.pos;
        auto indices = &sc->geometry.meshes.indices;
        auto normals = &sc->geometry.meshes.normals;
        auto uvs = &sc->geometry.meshes.uvs;
        const SharedVector<Mesh> &meshes = sc->geometry.meshes.meshes;

        const Spheres &spheres = sc->geometry.spheres;

        auto base_indices = (CUdeviceptr)indices->get_ptr();
        auto base_pos = (CUdeviceptr)pos->get_ptr();
        auto base_normals = (CUdeviceptr)normals->get_ptr();
        auto base_uvs = (CUdeviceptr)uvs->get_ptr();

        std::vector<PtHitGroupSbtRecord> hitgroup_records{};
        hitgroup_records.reserve(2 * (optixAS->num_meshes + optixAS->num_spheres));

        for (int i = 0; i < optixAS->num_meshes; i++) {
            PtHitGroupSbtRecord hg_rec{};
            auto &mesh = meshes[i];

            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg, &hg_rec));

            hg_rec.data.mesh.mesh_id = i;
            // TODO: these should be Mesh methods...
            hg_rec.data.mesh.pos = base_pos + mesh.pos_index * sizeof(vec3);
            hg_rec.data.mesh.indices = base_indices + mesh.indices_index * sizeof(u32);

            auto normals_index = mesh.normals_index;
            hg_rec.data.mesh.normals = base_normals + normals_index * sizeof(vec3);
            auto uvs_index = mesh.uvs_index;
            hg_rec.data.mesh.uvs = base_uvs + uvs_index * sizeof(vec2);

            hitgroup_records.emplace_back(hg_rec);
        }

        for (int i = 0; i < optixAS->num_spheres; i++) {
            PtHitGroupSbtRecord hg_rec{};
            OPTIX_CHECK(optixSbtRecordPackHeader(sphere_hitgroup_pg, &hg_rec));

            hg_rec.data.sphere.material_id = spheres.material_ids[i];
            hg_rec.data.sphere.light_id = spheres.light_ids[i];
            hg_rec.data.sphere.has_light = spheres.has_light[i];

            hitgroup_records.emplace_back(hg_rec);
        }

        size_t hitgroup_records_size =
            hitgroup_records.size() * sizeof(PtHitGroupSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_hitgroup_records),
                              hitgroup_records_size))

        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_hitgroup_records),
                              hitgroup_records.data(), hitgroup_records_size,
                              cudaMemcpyHostToDevice))

        sbt.raygenRecord = raygen_record.get_ptr();
        sbt.missRecordBase = miss_record.get_ptr();
        sbt.missRecordStrideInBytes = sizeof(PtMissSbtRecord);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = d_hitgroup_records;
        sbt.hitgroupRecordStrideInBytes = sizeof(PtHitGroupSbtRecord);
        sbt.hitgroupRecordCount = hitgroup_records.size();
    }
}

OptixRenderer::~OptixRenderer() {
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.hitgroupRecordBase)))

    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_pg));
    OPTIX_CHECK(optixProgramGroupDestroy(sphere_hitgroup_pg));
    OPTIX_CHECK(optixProgramGroupDestroy(miss_pg));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_pg));
    OPTIX_CHECK(optixModuleDestroy(module));
}
