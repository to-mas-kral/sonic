cmake_minimum_required(VERSION 3.27)
project(pt CUDA CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

#[[Main executable]]

add_executable(pt src/main.cu
        src/render_context_common.h
        src/render_context.cu
        src/framebuffer.cu
        src/framebuffer.h
        src/camera.h
        src/material.h
        src/light.h
        src/geometry/bvh.cu
        src/scene_loader.cu
        src/scene_loader.h
        src/envmap.h
        src/texture.h
        src/texture.cu

        src/utils/numtypes.h
        src/utils/shared_vector.h
        src/io/image_writer.h
        src/utils/cuda_err.h
        src/utils/rng.h
        src/utils/sampler.h

        src/io/progress_bar.h
        src/io/window.cu
        src/io/window.h

        src/kernels/megakernel.h
        src/kernels/megakernel.cu
        src/kernels/raygen.h
        src/kernels/optix_pt.h

        src/geometry/intersection.h
        src/geometry/axis.h
        src/geometry/aabb.h

        src/math/sampling.h
        src/math/vecmath.h
        src/math/math_utils.h

        src/optix_renderer.cu
        src/optix_renderer.h
        src/optix_common.h
        src/optix_as.h
        src/optix_common.cu

        src/integrator/utils.h

        src/geometry/intersection.h
        src/geometry/geometry.h
        src/geometry/geometry.cu
        src/geometry/bvh.h
)

link_directories(~/ManualPkgs/NVIDIA-OptiX-SDK-6.5.0-linux64/lib64/)
include_directories(~/ManualPkgs/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/include/)

set_target_properties(pt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

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

find_package(Stb REQUIRED)
target_include_directories(pt PRIVATE ${Stb_INCLUDE_DIR})

find_package(CLI11 CONFIG REQUIRED)
target_link_libraries(pt PRIVATE CLI11::CLI11)

find_package(pugixml CONFIG REQUIRED)
target_link_libraries(pt PRIVATE pugixml::pugixml)

find_package(spdlog CONFIG REQUIRED)
target_link_libraries(pt PRIVATE spdlog::spdlog)

find_package(minifb CONFIG REQUIRED)
target_link_libraries(pt PRIVATE minifb::minifb)

#[[OptiX modules]]

add_library(optix_pt OBJECT
        src/kernels/optix_pt.cu
)

target_link_libraries(optix_pt PRIVATE fmt::fmt)

set_property(TARGET optix_pt PROPERTY CUDA_PTX_COMPILATION ON)

set_target_properties(optix_pt PROPERTIES
        OUTPUT_NAME "optix_raycast.ptx"
)

target_compile_options(optix_pt PRIVATE
        -ptx -lineinfo --use_fast_math --relocatable-device-code=true --keep-device-functions
)

add_dependencies(pt optix_pt)

#[[Packaging]]

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCES_DIR}/)

set(resource_dir ${CMAKE_SOURCE_DIR}/resources)
set(package_dir ${CMAKE_BINARY_DIR}/package)

install(TARGETS pt DESTINATION ${package_dir})
install(FILES ${CMAKE_BINARY_DIR}/CMakeFiles/optix_pt.dir/src/kernels/optix_pt.ptx DESTINATION ${package_dir})
install(FILES ${CMAKE_BINARY_DIR}/CMakeFiles/optix_pt.dir/src/kernels/optix_pt.ptx DESTINATION ${CMAKE_BINARY_DIR})
install(DIRECTORY ${resource_dir} DESTINATION ${package_dir})
