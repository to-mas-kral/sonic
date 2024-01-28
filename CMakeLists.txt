cmake_minimum_required(VERSION 3.27)
project(pt CUDA CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

#[[Main executable]]

add_executable(pt
        src/main.cu
        src/render_context.h
        src/framebuffer.h
        src/camera.h
        src/materials/material.h
        src/emitter.h
        src/io/scene_loader.cu
        src/io/scene_loader.h
        src/envmap.h
        src/texture.h
        src/texture.cu
        src/scene.cu
        src/scene.h
        src/optix_renderer.cu
        src/optix_renderer.h
        src/optix_common.h
        src/optix_as.h
        src/optix_common.cu

        src/utils/basic_types.h
        src/utils/um_vector.h
        src/utils/cuda_err.h
        src/utils/sampler.h
        src/utils/algs.h

        src/io/image_writer.h
        src/io/progress_bar.h

        src/kernels/raygen.h
        src/kernels/optix_pt.h

        src/math/sampling.h
        src/math/vecmath.h
        src/math/math_utils.h
        src/math/transform.h

        src/integrator/utils.h
        src/integrator/light_sampler.cu
        src/integrator/light_sampler.h

        src/geometry/geometry.h
        src/geometry/geometry.cu

        src/scene/light.h

        src/color/rgb2spec.h
        src/color/sampled_spectrum.h
        src/color/cie_spectrums.h
        src/color/spectrum_consts.h
        src/color/color_space.h
        src/color/spectrum.cu
        src/color/spectral_data.h
        src/color/spectrum.h
        src/materials/plastic.h
        src/materials/common.h
        src/integrator/integrator_type.h
)

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

#[[OptiX modules]]

add_library(optix_pt OBJECT
        src/kernels/optix_pt.cu
)

target_link_libraries(optix_pt PRIVATE fmt::fmt)

set_property(TARGET optix_pt PROPERTY CUDA_PTX_COMPILATION ON)

target_compile_options(optix_pt PRIVATE
        -ptx -lineinfo --use_fast_math --relocatable-device-code=true --keep-device-functions
)

add_dependencies(pt optix_pt)

#[[Packaging]]

set(OPTIX_PT_OBJECT_FILE $<TARGET_OBJECTS:optix_pt>)

# Create a custom target to handle the post-build command
# Just having the copy command somehow doesn't work for OBJECT libraries...
add_custom_target(optix_pt_post_build
        COMMAND ${CMAKE_COMMAND} -E copy
        ${OPTIX_PT_OBJECT_FILE}
        ${CMAKE_BINARY_DIR}/optix_pt.ptx
        COMMENT "Copying optix_pt.cu to ${CMAKE_BINARY_DIR}/optix_pt.ptx"
        DEPENDS optix_pt
)

add_dependencies(pt optix_pt_post_build)

#[[Tests]]

add_executable(tests
        src/render_context.h
        src/framebuffer.h
        src/camera.h
        src/materials/material.h
        src/emitter.h
        src/io/scene_loader.cu
        src/io/scene_loader.h
        src/envmap.h
        src/texture.h
        src/texture.cu
        src/scene.cu
        src/scene.h

        src/utils/basic_types.h
        src/utils/um_vector.h
        src/utils/cuda_err.h
        src/utils/sampler.h
        src/utils/algs.h

        src/io/image_writer.h
        src/io/progress_bar.h

        src/kernels/raygen.h
        src/kernels/optix_pt.h

        src/math/sampling.h
        src/math/vecmath.h
        src/math/math_utils.h
        src/math/transform.h

        src/integrator/utils.h
        src/integrator/light_sampler.cu
        src/integrator/light_sampler.h

        src/geometry/geometry.h
        src/geometry/geometry.cu

        src/scene/light.h

        src/color/rgb2spec.h
        src/color/sampled_spectrum.h
        src/color/cie_spectrums.h
        src/color/spectrum_consts.h
        src/color/color_space.h
        src/color/spectrum.cu
        src/color/spectral_data.h
        src/color/spectrum.h

        src/utils/tests.cu
        src/materials/test_ggx.cu
)

find_package(Catch2 3 REQUIRED)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

target_link_libraries(tests PRIVATE pugixml::pugixml)

target_link_libraries(tests PRIVATE fmt::fmt)

target_link_libraries(tests PRIVATE unofficial::tinyexr::tinyexr)
