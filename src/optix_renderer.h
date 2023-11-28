#ifndef PT_OPTIX_RENDERER_H
#define PT_OPTIX_RENDERER_H

#include "kernels/optix_pt.h"
#include "optix_as.h"
#include "scene.h"
#include "utils/cuda_box.h"

class OptixRenderer {
public:
    OptixRenderer(Scene *sc, OptixDeviceContext context, OptixAS *optixAS);

    void
    launch(u32 width, u32 height) {
        OPTIX_CHECK(optixLaunch(pipeline, nullptr, launch_params.get_ptr(),
                                sizeof(PtParams), &sbt, width, height, 1));

        cudaDeviceSynchronize();
        CUDA_CHECK_LAST_ERROR();
    }

    void
    update_params(PtParams params) {
        params.gas_handle = optixAS->tlas_handle;
        launch_params.set(&params);
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
