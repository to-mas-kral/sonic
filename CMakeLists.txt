cmake_minimum_required(VERSION 3.27)
project(pt CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_75,-gencode arch=compute_75,code=sm_75")

add_executable(pt src/main.cu
        src/render_context.h
        src/framebuffer.h

        src/utils/numtypes.h
        src/utils/shared_vector.h
        src/utils/image_writer.h
        src/utils/cuda_err.h

        src/kernels/megakernel.h

        src/shapes/mesh.h
        src/shapes/triangle.h
)

target_compile_options(pt PRIVATE
        $<$<CONFIG:Debug>:-G> # Adds debugging symbols to CUDA code
)

find_package(fmt CONFIG REQUIRED)
target_link_libraries(pt PRIVATE fmt::fmt)

find_package(glm CONFIG REQUIRED)
target_link_libraries(pt PRIVATE glm::glm)

find_package(OpenEXR CONFIG REQUIRED)
target_link_libraries(pt PRIVATE OpenEXR::OpenEXR)

find_package(tinyexr CONFIG REQUIRED)
target_link_libraries(pt PRIVATE unofficial::tinyexr::tinyexr)

set_target_properties(pt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
