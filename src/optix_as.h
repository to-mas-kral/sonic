#ifndef PT_OPTIX_AS_H
#define PT_OPTIX_AS_H

#include "render_context_common.h"

class OptixAS {
public:
    // Have to use 1 build input per material
    void add_sphere_inputs(const Spheres &spheres,
                           std::vector<OptixBuildInput> &build_inputs,
                           const u32 *input_flags,
                           std::vector<CUdeviceptr> &spheres_d_centers,
                           std::vector<CUdeviceptr> &spheres_d_radiuses) {
        for (int i = 0; i < num_spheres; i++) {
            spheres_d_centers[i] = (CUdeviceptr)(spheres.centers.get_ptr() + i);
            spheres_d_radiuses[i] = (CUdeviceptr)(spheres.radiuses.get_ptr() + i);
        }

        for (int i = 0; i < num_spheres; i++) {
            OptixBuildInput bi{};
            bi.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;

            bi.sphereArray.numVertices = 1;
            bi.sphereArray.vertexBuffers = &spheres_d_centers[i];
            bi.sphereArray.radiusBuffers = &spheres_d_radiuses[i];

            bi.sphereArray.flags = input_flags;
            bi.sphereArray.numSbtRecords = 1;

            build_inputs.emplace_back(bi);
        }
    }

    void add_mesh_inputs(const RenderContext *rc, const SharedVector<Mesh> &meshes,
                         std::vector<OptixBuildInput> &build_inputs,
                         std::vector<CUdeviceptr> &mesh_d_poses,
                         std::vector<CUdeviceptr> &mesh_d_indices,
                         const u32 *triangle_input_flags) const {
        auto pos = &rc->geometry.meshes.pos;
        auto indices = &rc->geometry.meshes.indices;

        for (int i = 0; i < num_meshes; i++) {
            auto &mesh = meshes[i];
            mesh_d_poses[i] = (CUdeviceptr)pos->get_ptr() + mesh.pos_index * sizeof(vec3);
            mesh_d_indices[i] =
                (CUdeviceptr)indices->get_ptr() + mesh.indices_index * sizeof(u32);
        }

        for (int i = 0; i < num_meshes; i++) {
            auto &mesh = meshes[i];
            OptixBuildInput bi{};

            bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            bi.triangleArray.numVertices = mesh.num_vertices;
            bi.triangleArray.vertexBuffers = &mesh_d_poses[i];

            bi.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            bi.triangleArray.numIndexTriplets = mesh.num_indices / 3;
            bi.triangleArray.indexBuffer = mesh_d_indices[i];

            bi.triangleArray.flags = triangle_input_flags;
            bi.triangleArray.numSbtRecords = 1;

            build_inputs.emplace_back(bi);
        }
    }

    OptixTraversableHandle create_as(OptixDeviceContext context,
                                     const std::vector<OptixBuildInput> &build_inputs,
                                     CUdeviceptr *output_buffer) {
        OptixTraversableHandle gas_handle{};

        OptixAccelBuildOptions accel_options{};
        accel_options.buildFlags =
            OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        ;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gas_buffer_sizes{};
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options,
                                                 build_inputs.data(), build_inputs.size(),
                                                 &gas_buffer_sizes));

        f64 mbs = gas_buffer_sizes.outputSizeInBytes / 1024. / 1024.;
        spdlog::info("OptiX acceleration structure size is {} MBs before compaction",
                     mbs);

        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas),
                              gas_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(output_buffer),
                              gas_buffer_sizes.outputSizeInBytes));

        CUdeviceptr d_compacted_size;
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>((&d_compacted_size)), sizeof(u64)));

        OptixAccelEmitDesc property{};
        property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        property.result = d_compacted_size;

        OPTIX_CHECK(optixAccelBuild(
            context,
            nullptr, // CUDA stream
            &accel_options, build_inputs.data(), build_inputs.size(), d_temp_buffer_gas,
            gas_buffer_sizes.tempSizeInBytes, *output_buffer,
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

            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(*output_buffer)));
            *output_buffer = d_compacted_output_buffer;

            spdlog::info("OptiX acceleration structure compacted");
        }

        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));

        return gas_handle;
    }

    OptixAS(RenderContext *rc, OptixDeviceContext context) {
        /*
         * Create BLASes
         * */

        const uint32_t input_flags =
            OPTIX_GEOMETRY_FLAG_NONE | OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

        const SharedVector<Mesh> &meshes = rc->geometry.meshes.meshes;
        num_meshes = meshes.len();
        std::vector<OptixBuildInput> triangle_build_inputs{};
        triangle_build_inputs.reserve(num_meshes);

        std::vector<CUdeviceptr> mesh_d_poses = std::vector<CUdeviceptr>(num_meshes);
        std::vector<CUdeviceptr> mesh_d_indices = std::vector<CUdeviceptr>(num_meshes);

        add_mesh_inputs(rc, meshes, triangle_build_inputs, mesh_d_poses, mesh_d_indices,
                        &input_flags);
        triangle_as_handle =
            create_as(context, triangle_build_inputs, &triangle_as_output_buffer);

        const Spheres &spheres = rc->geometry.spheres;
        num_spheres = spheres.num_spheres;
        if (num_spheres > 0) {
            std::vector<OptixBuildInput> sphere_build_inputs{};
            sphere_build_inputs.reserve(num_spheres);

            std::vector<CUdeviceptr> spheres_d_centers =
                std::vector<CUdeviceptr>(num_spheres);
            std::vector<CUdeviceptr> spheres_d_radiuses =
                std::vector<CUdeviceptr>(num_spheres);

            add_sphere_inputs(spheres, sphere_build_inputs, &input_flags,
                              spheres_d_centers, spheres_d_radiuses);

            sphere_as_handle =
                create_as(context, sphere_build_inputs, &sphere_as_output_buffer);
        }

        /*
         * Create TLAS
         * */

        u32 num_instances = 1;
        std::array<OptixInstance, 2> instances{};
        float transform[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        memcpy(instances[0].transform, transform, sizeof(float) * 12);
        instances[0].instanceId = 0;
        instances[0].visibilityMask = 255;
        instances[0].sbtOffset = 0;
        instances[0].flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
        instances[0].traversableHandle = triangle_as_handle;

        if (num_spheres > 0) {
            num_instances++;

            memcpy(instances[1].transform, transform, sizeof(float) * 12);
            instances[1].instanceId = 1;
            instances[1].visibilityMask = 255;
            instances[1].sbtOffset = num_meshes;
            instances[1].flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
            instances[1].traversableHandle = sphere_as_handle;
        }

        void *d_instances;
        auto instances_size = sizeof(OptixInstance) * num_instances;
        CUDA_CHECK(cudaMalloc(&d_instances, instances_size));
        CUDA_CHECK(cudaMemcpy(d_instances, instances.data(), instances_size,
                              cudaMemcpyHostToDevice));

        std::vector<OptixBuildInput> tlas_build_input(1);
        tlas_build_input[0].type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        tlas_build_input[0].instanceArray.instances =
            reinterpret_cast<CUdeviceptr>(d_instances);
        tlas_build_input[0].instanceArray.numInstances = num_instances;

        tlas_handle = create_as(context, tlas_build_input, &tlas_output_buffer);
    }

    ~OptixAS() {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(triangle_as_output_buffer)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sphere_as_output_buffer)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(tlas_output_buffer)));
    }

    OptixTraversableHandle tlas_handle{};
    u32 num_meshes;
    u32 num_spheres;

private:
    OptixTraversableHandle triangle_as_handle{};
    OptixTraversableHandle sphere_as_handle{};
    CUdeviceptr triangle_as_output_buffer{};
    CUdeviceptr sphere_as_output_buffer{};
    CUdeviceptr tlas_output_buffer{};
};

#endif // PT_OPTIX_AS_H
