#ifndef PT_OPTIX_PT_H
#define PT_OPTIX_PT_H

#include <optix_types.h>

#include "../optix_common.h"
#include "../render_context_common.h"
#include "../utils/numtypes.h"

struct OpIntersection {
    u32 material_id;
    u32 light_id;
    bool has_light;
    vec3 normal;
    vec3 pos;
    vec2 uv;
};

struct PtParams {
    u32 sample_index;
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

enum class ShapeType {
    Mesh = 0,
    Sphere = 1,
};

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
