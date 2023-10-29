
#include "optix_triangle.h"

#include <optix_device.h>

#include "../geometry/wavefront_ray.h"
#include "../utils/numtypes.h"

extern "C" {
__constant__ Params params;
}

/*static __forceinline__ __device__ void setPayload(float3 p) {
    optixSetPayload_0(__float_as_uint(p.x));
    optixSetPayload_1(__float_as_uint(p.y));
    optixSetPayload_2(__float_as_uint(p.z));
}*/

extern "C" __global__ void __raygen__rg() {
    // Lookup our location within the launch grid
    /*
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const vec2 st = vec2(static_cast<float>(idx.x) / static_cast<float>(dim.x),
                         static_cast<float>(idx.y) / static_cast<float>(dim.y));

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    Ray ray = params.cam->get_ray(st.x, st.y);


    float3 rayorig = make_float3(ray.o.x, ray.o.y, ray.o.z);
    float3 raydir = make_float3(ray.dir.x, ray.dir.y, ray.dir.z);

    int a = 10;

    // Trace the ray against our scene hierarchy
    unsigned int p0, p1, p2;
    optixTrace(params.gas_handle, rayorig, raydir,
               0.0f,                     // Min intersection distance
               1e16f,                    // Max intersection distance
               0.0f,                     // rayTime -- used for motion blur
               OptixVisibilityMask(255), // Specify always visible
               OPTIX_RAY_FLAG_NONE,
               0, // SBT offset   -- See SBT discussion
               1, // SBT stride   -- See SBT discussion
               0, // missSBTIndex -- See SBT discussion
               p0, p1, p2);

    vec3 result;
    result.x = __uint_as_float(p0);
    result.y = __uint_as_float(p1);
    result.z = __uint_as_float(p2);

    */
}

extern "C" __global__ void __miss__ms() {
    //MissData *miss_data = reinterpret_cast<MissData *>(optixGetSbtDataPointer());
    //setPayload(miss_data->bg_color);
}

extern "C" __global__ void __closesthit__ch() {
    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indlucing barycentric coordinates.
    /*const float2 barycentrics = optixGetTriangleBarycentrics();
    const auto prim_index = optixGetPrimitiveIndex();

    setPayload(make_float3(barycentrics.x, barycentrics.y, 1.0f));*/
}


/*
     * Initialize OptiX API
     * */
// Initialize CUDA with a no-op call to the the CUDA runtime API
/*cudaFree(nullptr);

// Initialize the OptiX API, loading all API entry points
OPTIX_CHECK(optixInit());

// Specify options for this context. We will use the default options.
OptixDeviceContextOptions options = {};

// Associate a CUDA context (and therefore a specific GPU) with this
// device context
CUcontext cuCtx = nullptr; // NULL means take the current active context

OptixDeviceContext context = nullptr;
OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

// Specify options for the build. We use default options for simplicity.
OptixAccelBuildOptions accel_options = {};
accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

// Triangle build input: simple list of three vertices
const std::array<float3, 3> vertices = {
    {{-0.5f, -0.5f, 0.0f}, {0.5f, -0.5f, 0.0f}, {0.0f, 0.5f, 0.0f}}};

// Allocate and copy device memory for our input triangle vertices
const size_t vertices_size = sizeof(float3) * vertices.size();
CUdeviceptr d_vertices = 0;

CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_vertices), vertices_size));
CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_vertices), vertices.data(),
                      vertices_size, cudaMemcpyHostToDevice));

// Populate the build input struct with our triangle data as well as
// information about the sizes and types of our data
const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
OptixBuildInput triangle_input = {};
triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
triangle_input.triangleArray.numVertices = vertices.size();
triangle_input.triangleArray.vertexBuffers = &d_vertices;
// TODO: index buffer
triangle_input.triangleArray.flags = triangle_input_flags;
triangle_input.triangleArray.numSbtRecords = 1;

// Query OptiX for the memory requirements for our GAS
OptixAccelBufferSizes gas_buffer_sizes;
OPTIX_CHECK(
    optixAccelComputeMemoryUsage(context, // The device context we are using
                                 &accel_options,
                                 &triangle_input, // Describes our geometry
                                 1, // Number of build inputs, could have multiple
                                 &gas_buffer_sizes));

// Allocate device memory for the scratch space buffer as well
// as the GAS itself
CUdeviceptr d_temp_buffer_gas;
CUdeviceptr d_gas_output_buffer;
CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas),
                      gas_buffer_sizes.tempSizeInBytes));
CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_gas_output_buffer),
                      gas_buffer_sizes.outputSizeInBytes));

// Now build the GAS
OptixTraversableHandle gas_handle{};
OPTIX_CHECK(optixAccelBuild(context,
                            nullptr, // CUDA stream
                            &accel_options, &triangle_input,
                            1, // num build inputs
                            d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes,
                            d_gas_output_buffer, gas_buffer_sizes.outputSizeInBytes,
                            &gas_handle, // Output handle to the struct
                            nullptr,     // emitted property list
                            0));         // num emitted properties

// TODO: compaction
// optixAccelCompact(context,
//                  nullptr,
//                  gas_handle,
//                  d_gas_output_buffer,
//                  gas_buffer_sizes.outputSizeInBytes,
//                  OptixTraversableHandle* outputHandle);

// We can now free scratch space used during the build
CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));

// Pipeline options must be consistent for all modules used in a
// single pipeline
OptixPipelineCompileOptions pipeline_compile_options = {};
pipeline_compile_options.usesMotionBlur = false;

// This option is important to ensure we compile code which is optimal
// for our scene hierarchy. We use a single GAS – no instancing or
// multi-level hierarchies
pipeline_compile_options.traversableGraphFlags =
    OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

// Our device code uses 3 payload registers (r,g,b output value)
pipeline_compile_options.numPayloadValues = 3;

// This is the name of the param struct variable in our device code
pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

OptixModule module = nullptr; // The output module
{
    OptixModuleCompileOptions module_compile_options = {};
    const std::string ptx = read_file("optix_triangle.ptx");

    OPTIX_CHECK_LOG(optixModuleCreate(context, &module_compile_options,
                                      &pipeline_compile_options, ptx.c_str(),
                                      ptx.size(), LOG, &LOG_SIZE, &module));
}

//
// Create program groups
//
OptixProgramGroup raygen_prog_group = nullptr;
OptixProgramGroup miss_prog_group = nullptr;
OptixProgramGroup hitgroup_prog_group = nullptr;
{
    OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc = {}; //
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &raygen_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, LOG, &LOG_SIZE,
                                            &raygen_prog_group));

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &miss_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, LOG, &LOG_SIZE,
                                            &miss_prog_group));

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &hitgroup_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, LOG, &LOG_SIZE,
                                            &hitgroup_prog_group));
}

//
// Link pipeline
//
OptixPipeline pipeline = nullptr;
{
    const uint32_t max_trace_depth = 1;
    OptixProgramGroup program_groups[] = {raygen_prog_group, miss_prog_group,
                                          hitgroup_prog_group};

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    OPTIX_CHECK_LOG(optixPipelineCreate(
        context, &pipeline_compile_options, &pipeline_link_options, program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]), LOG, &LOG_SIZE,
        &pipeline));

    OptixStackSizes stack_sizes = {};
    for (auto &prog_group : program_groups) {
        OPTIX_CHECK(
            optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                           0, // maxCCDepth
                                           0, // maxDCDEpth
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state,
                                           &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline, direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state, continuation_stack_size,
        1 // maxTraversableDepth
        ));
}

//
// Set up shader binding table
//
OptixShaderBindingTable sbt = {};
{
    CUdeviceptr raygen_record;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&raygen_record), raygen_record_size));
    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(raygen_record), &rg_sbt,
                          raygen_record_size, cudaMemcpyHostToDevice));

    CUdeviceptr miss_record;
    size_t miss_record_size = sizeof(MissSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));
    MissSbtRecord ms_sbt;
    ms_sbt.data = {0.3f, 0.1f, 0.2f};
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(miss_record), &ms_sbt,
                          miss_record_size, cudaMemcpyHostToDevice));

    CUdeviceptr hitgroup_record;
    size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record),
                          hitgroup_record_size));
    HitGroupSbtRecord hg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(hitgroup_record), &hg_sbt,
                          hitgroup_record_size, cudaMemcpyHostToDevice));

    sbt.raygenRecord = raygen_record;
    sbt.missRecordBase = miss_record;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount = 1;
}

//
// launch
//
{
    Params params{};
    params.rc = rc;
    params.fb = &rc->get_framebuffer();
    params.cam = &rc->get_cam();
    params.gas_handle = gas_handle;

    CUdeviceptr d_param;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_param), &params, sizeof(params),
                          cudaMemcpyHostToDevice));

        OPTIX_CHECK(optixLaunch(pipeline, nullptr, d_param, sizeof(Params), &sbt,
                                attribs.resy, attribs.resx, */
        /*depth =*//*1));

    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR();

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_param)));
    }

    cudaDeviceSynchronize();

    ImageWriter::write_framebuffer("ptout_optix.exr", rc->get_framebuffer(), 1);

    //
    // Cleanup
    //
    {
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_gas_output_buffer)));

    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
    OPTIX_CHECK(optixModuleDestroy(module));

    OPTIX_CHECK(optixDeviceContextDestroy(context));
    }*/
