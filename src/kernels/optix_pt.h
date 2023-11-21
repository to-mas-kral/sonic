#ifndef PT_OPTIX_PT_H
#define PT_OPTIX_PT_H

#include <optix_types.h>

#include "../optix_common.h"
#include "../render_context.h"
#include "../utils/basic_types.h"

struct PtParams {
    Light *lights;
    Texture *textures;
    Material *materials;
    Mesh *meshes;
    Framebuffer *fb;
    RenderContext *rc;
    OptixTraversableHandle gas_handle;
};

struct PtRayGenData {};

struct PtMissData {};

struct PtHitGroupData {
    // Shape type can be queried by optixGetPrimitiveType()
    union {
        struct {
            u32 mesh_id;
            CUdeviceptr pos;
            CUdeviceptr indices;
            CUdeviceptr normals;
            CUdeviceptr uvs;
        } mesh{};

        struct {
            u32 material_id;
            u32 light_id;
            bool has_light = false;
        } sphere;
    };
};

typedef SbtRecord<PtRayGenData> PtRayGenSbtRecord;
typedef SbtRecord<PtMissData> PtMissSbtRecord;
typedef SbtRecord<PtHitGroupData> PtHitGroupSbtRecord;

#endif // PT_OPTIX_PT_H
