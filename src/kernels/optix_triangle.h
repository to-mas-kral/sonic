#ifndef PT_OPTIX_TRIANGLE_H
#define PT_OPTIX_TRIANGLE_H

/*#include "../render_context_common.h"*/

struct Params {
    /*RenderContext *rc;*/
    OptixTraversableHandle gas_handle;
};

struct RayGenData {
    // No data needed
};

struct MissData {
    // No data needed
};

struct HitGroupData {
    // No data needed
};

template <typename T> struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

#endif // PT_OPTIX_TRIANGLE_H
