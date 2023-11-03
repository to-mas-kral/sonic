#include <array>
#include <vector>

#include "optix_common.h"
#include "optix_renderer.h"
#include "utils/cuda_box.h"

OptixRenderer::OptixRenderer(RenderContext *rc, OptixDeviceContext context,
                             OptixAS *optixAS)
    : optixAS(optixAS) {
    auto pipeline_compile_options = make_optix_pipeline_compile_options(9, "params");
    make_optix_module(context, &pipeline_compile_options, &module, "optix_pt.ptx");

    OptixProgramGroupOptions pg_options = {};
    make_optix_program_groups(context, module, &pg_options, &raygen_pg, &miss_pg,
                              &hitgroup_pg);

    const u32 max_trace_depth = 1;
    std::array<OptixProgramGroup, 3> pgs = {raygen_pg, miss_pg, hitgroup_pg};
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

        std::vector<PtHitGroupSbtRecord> hitgroup_records(optixAS->num_meshes);
        for (int i = 0; i < optixAS->num_meshes; i++) {
            PtHitGroupSbtRecord hg_rec;

            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg, &hg_rec));
            hg_rec.data.mesh_id = i;
            hg_rec.data.pos = optixAS->mesh_d_poses[i];
            hg_rec.data.indices = optixAS->mesh_d_indices[i];
            hitgroup_records[i] = hg_rec;
        }

        size_t hitgroup_records_size =
            hitgroup_records.size() * sizeof(PtHitGroupSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_hitgroup_records),
                              hitgroup_records_size));

        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_hitgroup_records),
                              hitgroup_records.data(), hitgroup_records_size,
                              cudaMemcpyHostToDevice));

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
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.hitgroupRecordBase)));

    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_pg));
    OPTIX_CHECK(optixProgramGroupDestroy(miss_pg));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_pg));
    OPTIX_CHECK(optixModuleDestroy(module));
}
