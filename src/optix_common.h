#ifndef PT_OPTIX_COMMON_H
#define PT_OPTIX_COMMON_H

#include <fstream>
#include <optix_host.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <fmt/core.h>

#include "utils/cuda_err.h"
#include "utils/numtypes.h"

auto read_file(std::string_view path) -> std::string;

template <typename T> struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

static void context_log_cb(unsigned int level, const char *tag, const char *message,
                           void *);

OptixDeviceContext init_optix();

OptixPipelineCompileOptions make_optix_pipeline_compile_options(int num_payload_values,
                                                                const char *params_name);

void make_optix_module(OptixDeviceContext context,
                       const OptixPipelineCompileOptions *pipeline_compile_options,
                       OptixModule *module, const char *filepath,
                       OptixModuleCompileOptions *module_compile_options);

template <size_t N>
void link_optix_pipeline(OptixDeviceContext context,
                         const OptixPipelineCompileOptions *pipeline_compile_options,
                         u32 max_trace_depth,
                         const std::array<OptixProgramGroup, N> &program_groups,
                         OptixPipeline *pipeline) {
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    OPTIX_CHECK_LOG(optixPipelineCreate(context, pipeline_compile_options,
                                        &pipeline_link_options, program_groups.data(), N,
                                        LOG, &LOG_SIZE, pipeline));

    OptixStackSizes stack_sizes = {};
    for (auto &prog_group : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, *pipeline));
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
        *pipeline, direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state, continuation_stack_size,
        2 // maxTraversableDepth
        ));
}

void make_optix_program_groups(OptixDeviceContext context, OptixModule module,
                               OptixModule sphere_is_module,
                               const OptixProgramGroupOptions *pg_options,
                               OptixProgramGroup *raygen_pg, OptixProgramGroup *miss_pg,
                               OptixProgramGroup *hitgroup_pg,
                               OptixProgramGroup *sphere_hitgroup_pg);

#endif // PT_OPTIX_COMMON_H
