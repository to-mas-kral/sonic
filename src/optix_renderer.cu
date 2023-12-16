#include <array>
#include <vector>

#include "optix_common.h"
#include "optix_renderer.h"

OptixRenderer::OptixRenderer(Scene *sc, OptixDeviceContext context, OptixAS *optixAS)
    : optixAS(optixAS) {
    auto pipeline_compile_options = make_optix_pipeline_compile_options(5, "params");

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

        std::vector<PtHitGroupSbtRecord> hitgroup_records{};
        hitgroup_records.reserve(2 * (optixAS->num_meshes + optixAS->num_spheres));

        for (int i = 0; i < optixAS->num_meshes; i++) {
            PtHitGroupSbtRecord hg_rec{};

            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg, &hg_rec));

            hg_rec.data.mesh.mesh_id = i;

            hitgroup_records.emplace_back(hg_rec);
        }

        for (int i = 0; i < optixAS->num_spheres; i++) {
            PtHitGroupSbtRecord hg_rec{};
            OPTIX_CHECK(optixSbtRecordPackHeader(sphere_hitgroup_pg, &hg_rec));

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
