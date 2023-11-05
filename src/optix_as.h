#ifndef PT_OPTIX_AS_H
#define PT_OPTIX_AS_H

#include "render_context_common.h"

class OptixAS {
public:
    OptixAS(RenderContext *rc, OptixDeviceContext_t *context) {
        auto pos = &rc->get_pos();
        auto indices = &rc->get_indices();
        const SharedVector<Mesh> &meshes = rc->get_meshes();
        num_meshes = meshes.len();

        std::vector<CUdeviceptr> mesh_d_poses = std::vector<CUdeviceptr>(num_meshes);
        std::vector<CUdeviceptr> mesh_d_indices = std::vector<CUdeviceptr>(num_meshes);

        for (int i = 0; i < num_meshes; i++) {
            auto &mesh = meshes[i];
            mesh_d_poses[i] = (CUdeviceptr)pos->get_ptr() + mesh.pos_index * sizeof(vec3);
            mesh_d_indices[i] =
                (CUdeviceptr)indices->get_ptr() + mesh.indices_index * sizeof(u32);
        }

        std::vector<OptixBuildInput> triangle_inputs(num_meshes);
        const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

        for (int i = 0; i < num_meshes; i++) {
            auto &mesh = meshes[i];
            triangle_inputs[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            triangle_inputs[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_inputs[i].triangleArray.numVertices = mesh.num_vertices;
            triangle_inputs[i].triangleArray.vertexBuffers = &mesh_d_poses[i];

            triangle_inputs[i].triangleArray.indexFormat =
                OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangle_inputs[i].triangleArray.numIndexTriplets = mesh.num_indices / 3;
            triangle_inputs[i].triangleArray.indexBuffer = mesh_d_indices[i];

            triangle_inputs[i].triangleArray.flags = triangle_input_flags;
            triangle_inputs[i].triangleArray.numSbtRecords = 1;
        }

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags =
            OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        ;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(
            optixAccelComputeMemoryUsage(context, &accel_options, triangle_inputs.data(),
                                         triangle_inputs.size(), &gas_buffer_sizes));

        f64 mbs = gas_buffer_sizes.outputSizeInBytes / 1024. / 1024.;
        spdlog::info("OptiX acceleration structure size is {} MBs before compaction",
                     mbs);

        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas),
                              gas_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_gas_output_buffer),
                              gas_buffer_sizes.outputSizeInBytes));

        CUdeviceptr d_compacted_size;
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>((&d_compacted_size)), sizeof(u64)));

        OptixAccelEmitDesc property = {};
        property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        property.result = d_compacted_size;

        OPTIX_CHECK(optixAccelBuild(
            context,
            nullptr, // CUDA stream
            &accel_options, triangle_inputs.data(), triangle_inputs.size(),
            d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, d_gas_output_buffer,
            gas_buffer_sizes.outputSizeInBytes, &gas_handle, &property, 1));

        size_t compacted_size;
        CUDA_CHECK(cudaMemcpy(&compacted_size, reinterpret_cast<void *>(d_compacted_size),
                              sizeof(size_t), cudaMemcpyDeviceToHost));

        f64 mbs2 = compacted_size / 1024. / 1024.;
        spdlog::info("OptiX acceleration structure size after compaction is {} MBs",
                     mbs2);

        if (compacted_size < gas_buffer_sizes.outputSizeInBytes) {
            spdlog::info("Compacting OptiX acceleration structure");
            CUdeviceptr d_compacted_output_buffer;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_compacted_output_buffer),
                                  compacted_size));

            OPTIX_CHECK(optixAccelCompact(context, nullptr, gas_handle,
                                          d_compacted_output_buffer, compacted_size,
                                          &gas_handle));

            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_gas_output_buffer)));
            d_gas_output_buffer = d_compacted_output_buffer;

            spdlog::info("OptiX acceleration structure compacted");
        }

        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));
    }

    ~OptixAS() { CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_gas_output_buffer))); }

    OptixTraversableHandle gas_handle{};
    u32 num_meshes = 0;

private:
    CUdeviceptr d_gas_output_buffer{};
};

#endif // PT_OPTIX_AS_H
