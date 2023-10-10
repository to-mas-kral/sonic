cmake_minimum_required(VERSION 3.27)
project(pt CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_75,-gencode arch=compute_75,code=sm_75")

add_executable(pt src/main.cu
        src/render_context.h
        src/framebuffer.h
        src/camera.h
        src/material.h
        src/light.h

        src/utils/numtypes.h
        src/utils/shared_vector.h
        src/utils/image_writer.h
        src/utils/cuda_err.h
        src/utils/rng.h

        src/kernels/megakernel.h

        src/shapes/mesh.h
        src/shapes/triangle.h
        src/geometry/intersection.h

        src/math/sampling.h
        src/math/vecmath.h
        src/math/math_utils.h
)

target_compile_options(pt PRIVATE
        $<$<CONFIG:Debug>:-G> # Adds debugging symbols to CUDA code
)

target_compile_options(pt PRIVATE
        $<$<CONFIG:Release>:-lineinfo> # Adds line info to optimized code
)

find_package(fmt CONFIG REQUIRED)
target_link_libraries(pt PRIVATE fmt::fmt)

find_package(glm CONFIG REQUIRED)
target_link_libraries(pt PRIVATE glm::glm)

find_package(tinyexr CONFIG REQUIRED)
target_link_libraries(pt PRIVATE unofficial::tinyexr::tinyexr)

find_package(CLI11 CONFIG REQUIRED)
target_link_libraries(pt PRIVATE CLI11::CLI11)

set_target_properties(pt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
