#ifndef PT_OPTIX_PT_H
#define PT_OPTIX_PT_H

#include <optix_types.h>

#include "../integrator/integrator_type.h"
#include "../optix_common.h"
#include "../render_context.h"
#include "../utils/basic_types.h"

// TODO: convert to const
struct PtParams {
    Light *lights;
    Texture *textures;
    Material *materials;
    Mesh *meshes;
    Framebuffer *fb;
    RenderContext *rc;
    OptixTraversableHandle gas_handle;
    mat4 cam_to_world;
    u32 frame;
    u32 max_depth;
    IntegratorType integrator_type;
};

struct PtRayGenData {};

struct PtMissData {};

struct PtHitGroupData {
    // Shape type can be queried by optixGetPrimitiveType()
    union {
        struct {
            u32 mesh_id;
        } mesh{};

        struct {
        } sphere;
    };
};

typedef SbtRecord<PtRayGenData> PtRayGenSbtRecord;
typedef SbtRecord<PtMissData> PtMissSbtRecord;
typedef SbtRecord<PtHitGroupData> PtHitGroupSbtRecord;

#endif // PT_OPTIX_PT_H
