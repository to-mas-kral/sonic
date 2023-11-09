#ifndef PT_OPTIX_RENDERER_H
#define PT_OPTIX_RENDERER_H

#include "kernels/optix_pt.h"
#include "optix_as.h"
#include "render_context_common.h"
#include "utils/cuda_box.h"

class OptixRenderer {
public:
    OptixRenderer(RenderContext *rc, OptixDeviceContext context, OptixAS *optixAS);

    void launch(PtParams params, u32 width, u32 height) {
        params.gas_handle = optixAS->tlas_handle;
        launch_params.set(&params);
        OPTIX_CHECK(optixLaunch(pipeline, nullptr, launch_params.get_ptr(),
                                sizeof(PtParams), &sbt, width, height, 1));

        cudaDeviceSynchronize();
        CUDA_CHECK_LAST_ERROR();
    }

    ~OptixRenderer();

private:
    CudaBox<PtParams> launch_params{};

    OptixAS *optixAS;

    OptixShaderBindingTable sbt{};
    CUdeviceptr d_hitgroup_records{};

    CudaBox<PtRayGenSbtRecord> raygen_record{};
    CudaBox<PtMissSbtRecord> miss_record{};

    OptixPipeline pipeline{};
    OptixModule module{};

    OptixProgramGroup raygen_pg{};
    OptixProgramGroup miss_pg{};
    OptixProgramGroup hitgroup_pg{};
    OptixProgramGroup sphere_hitgroup_pg{};
};

#endif // PT_OPTIX_RENDERER_H
