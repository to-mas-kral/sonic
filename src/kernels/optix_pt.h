#ifndef PT_OPTIX_PT_H
#define PT_OPTIX_PT_H

#include <optix_types.h>

#include "../optix_common.h"
#include "../render_context_common.h"
#include "../utils/numtypes.h"

struct PtParams {
    Light *lights;
    Material *materials;
    Mesh *meshes;
    Framebuffer *fb;
    RenderContext *rc;
    OptixTraversableHandle gas_handle;
};

struct PtRayGenData {};

struct PtMissData {};

struct PtHitGroupData {
    u32 mesh_id;
    CUdeviceptr pos;
    CUdeviceptr indices;
};

typedef SbtRecord<PtRayGenData> PtRayGenSbtRecord;
typedef SbtRecord<PtMissData> PtMissSbtRecord;
typedef SbtRecord<PtHitGroupData> PtHitGroupSbtRecord;

#endif // PT_OPTIX_PT_H
