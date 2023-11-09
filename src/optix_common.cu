
#include "optix_common.h"
#include "optix_types.h"
#include <spdlog/spdlog.h>

auto read_file(std::string_view path) -> std::string {
    constexpr auto read_size = std::size_t(4096);
    auto stream = std::ifstream(path.data(), std::ios::binary);
    stream.exceptions(std::ios_base::badbit);

    if (!stream) {
        throw std::ios_base::failure("file does not exist");
    }

    auto out = std::string();
    auto buf = std::string(read_size, '\0');
    while (stream.read(&buf[0], read_size)) {
        out.append(buf, 0, stream.gcount());
    }
    out.append(buf, 0, stream.gcount());
    return out;
}

void context_log_cb(unsigned int level, const char *tag, const char *message, void *) {
    spdlog::warn("{} {} {}", (int)level, tag, message);
}

OptixDeviceContext init_optix() {
    OptixDeviceContext context = nullptr;

    cudaFree(nullptr);
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
#ifndef NDEBUG
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif

    OPTIX_CHECK(optixDeviceContextCreate(nullptr, &options, &context));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(context, context_log_cb, nullptr, 3));

    return context;
}

OptixPipelineCompileOptions make_optix_pipeline_compile_options(int num_payload_values,
                                                                const char *params_name) {
    // Pipeline options must be consistent for all modules used in a
    // single pipeline
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;

    pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;

    pipeline_compile_options.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;

    pipeline_compile_options.numPayloadValues = num_payload_values;
    pipeline_compile_options.pipelineLaunchParamsVariableName = params_name;

    return pipeline_compile_options;
}

void make_optix_module(OptixDeviceContext context,
                       const OptixPipelineCompileOptions *pipeline_compile_options,
                       OptixModule *module, const char *filepath,
                       OptixModuleCompileOptions *module_compile_options) {
#ifndef NDEBUG
    module_compile_options->debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    module_compile_options->optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
#endif

    const std::string ptx = read_file(filepath);

    OPTIX_CHECK_LOG(optixModuleCreate(context, module_compile_options,
                                      pipeline_compile_options, ptx.c_str(), ptx.size(),
                                      LOG, &LOG_SIZE, module));
}

void make_optix_program_groups(OptixDeviceContext context, OptixModule module,
                               OptixModule sphere_is_module,
                               const OptixProgramGroupOptions *pg_options,
                               OptixProgramGroup *raygen_pg, OptixProgramGroup *miss_pg,
                               OptixProgramGroup *hitgroup_pg,
                               OptixProgramGroup *sphere_hitgroup_pg) {
    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &raygen_prog_group_desc,
                                            1, // num program groups
                                            pg_options, LOG, &LOG_SIZE, raygen_pg));

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &miss_prog_group_desc,
                                            1, // num program groups
                                            pg_options, LOG, &LOG_SIZE, miss_pg));

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &hitgroup_prog_group_desc,
                                            1, // num program groups
                                            pg_options, LOG, &LOG_SIZE, hitgroup_pg));

    OptixProgramGroupDesc sphere_hitgroup_prog_group_desc = {};
    sphere_hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    sphere_hitgroup_prog_group_desc.hitgroup.moduleIS = sphere_is_module;
    sphere_hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr;
    sphere_hitgroup_prog_group_desc.hitgroup.moduleCH = module;
    sphere_hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &sphere_hitgroup_prog_group_desc,
                                            1, // num program groups
                                            pg_options, LOG, &LOG_SIZE,
                                            sphere_hitgroup_pg));
}
