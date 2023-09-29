cmake_minimum_required(VERSION 3.27)
project(pt CUDA)

set(CMAKE_CUDA_STANDARD 17)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")

add_executable(pt src/main.cu
        src/vec3.h
        src/ray.h
        src/hitable.h
        src/hitable_list.h
        src/sphere.h
        src/camera.h
        src/mesh.h
        src/numtypes.h
        src/utils/shared_vector.h
        src/render_context.h
        src/kernels/megakernel.h
)

find_package(fmt CONFIG REQUIRED)
target_link_libraries(pt PRIVATE fmt::fmt)

find_package(glm CONFIG REQUIRED)
target_link_libraries(pt PRIVATE glm::glm)

set_target_properties(pt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
