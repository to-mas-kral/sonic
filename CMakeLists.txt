cmake_minimum_required(VERSION 3.27)
project(pt CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#[[Main executable]]

add_library(pt_lib
        src/render_context.h
        src/framebuffer.h
        src/camera.h
        src/embree_device.h

        src/scene/emitter.h
        src/scene/envmap.h
        src/scene/texture.h
        src/scene/texture.cpp
        src/scene/scene.cpp
        src/scene/scene.h
        src/scene/light.h
        src/scene/emitter.cpp
        src/scene/envmap.cpp

        src/utils/basic_types.h
        src/utils/sampler.h
        src/utils/algs.h
        src/utils/chunk_allocator.h
        src/utils/render_threads.h
        src/utils/render_threads.cpp

        src/io/scene_loader.cpp
        src/io/scene_loader.h
        src/io/image_writer.h
        src/io/progress_bar.h

        src/math/sampling.h
        src/math/vecmath.h
        src/math/math_utils.h
        src/math/transform.h
        src/math/piecewise_dist.cpp
        src/math/sampling.cpp
        src/math/transform.cpp

        src/integrator/integrator.h
        src/integrator/utils.h
        src/integrator/light_sampler.cpp
        src/integrator/light_sampler.h
        src/integrator/integrator_type.h
        src/integrator/mis_nee_integrator.cpp
        src/integrator/intersection.h

        src/geometry/geometry.h
        src/geometry/geometry.cpp

        src/color/rgb2spec.h
        src/color/rgb2spec.cpp
        src/color/sampled_spectrum.h
        src/color/sampled_spectrum.cpp
        src/color/cie_spectrums.h
        src/color/spectrum_consts.h
        src/color/color_space.h
        src/color/spectrum.cpp
        src/color/spectral_data.h
        src/color/spectrum.h

        src/materials/material.h
        src/materials/material.cpp
        src/materials/plastic.h
        src/materials/plastic.cpp
        src/materials/common.h
        src/materials/common.cpp
        src/materials/diffuse.h
        src/materials/diffuse.cpp
        src/materials/dielectric.h
        src/materials/dielectric.cpp
        src/materials/conductor.h
        src/materials/conductor.cpp
        src/materials/rough_conductor.h
        src/materials/rough_conductor.cpp
        src/materials/trowbridge_reitz_ggx.h
        src/materials/trowbridge_reitz_ggx.cpp
        src/materials/rough_plastic.h
        src/materials/rough_plastic.cpp
        src/utils/sampler.cpp
        src/settings.h
        src/pbrt_loader/pbrt_loader.cpp
        src/pbrt_loader/pbrt_loader.h
        src/pbrt_loader/lexer.cpp
        src/pbrt_loader/lexer.h
        src/scene/scene_attribs.h
        src/pbrt_loader/param.h
)

target_compile_options(pt_lib PUBLIC
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)

target_link_options(pt_lib PUBLIC
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)

find_package(fmt CONFIG REQUIRED)
target_link_libraries(pt_lib PUBLIC fmt::fmt)

find_package(tinyexr CONFIG REQUIRED)
target_link_libraries(pt_lib PUBLIC unofficial::tinyexr::tinyexr)

find_package(Stb REQUIRED)
target_include_directories(pt_lib PUBLIC ${Stb_INCLUDE_DIR})

find_package(CLI11 CONFIG REQUIRED)
target_link_libraries(pt_lib PUBLIC CLI11::CLI11)

find_package(pugixml CONFIG REQUIRED)
target_link_libraries(pt_lib PUBLIC pugixml::pugixml)

find_package(spdlog CONFIG REQUIRED)
target_link_libraries(pt_lib PUBLIC spdlog::spdlog)

find_package(embree 4 REQUIRED)
target_link_libraries(pt_lib PUBLIC embree)


add_executable(pt
        src/main.cpp
)

target_link_libraries(pt PRIVATE pt_lib)

#[[Packaging]]

add_subdirectory(${CMAKE_SOURCE_DIR}/tools/rgb2spec)

add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/rgb2spec.out
        COMMAND ${CMAKE_BINARY_DIR}/tools/rgb2spec/rgb2spec_opt 64 ${CMAKE_BINARY_DIR}/rgb2spec.out sRGB
        COMMENT "Creating spectral uplifting file rgb2spec.out"
        DEPENDS rgb2spec_opt
)

add_custom_target(
        run_rgb2spec_opt
        DEPENDS ${CMAKE_BINARY_DIR}/rgb2spec.out
)

add_dependencies(pt run_rgb2spec_opt)

#[[Tests]]
add_executable(tests
        src/pbrt_loader/test_lexer.cpp
        src/pbrt_loader/test_params_parser.cpp
)

target_link_libraries(tests PRIVATE pt_lib)

find_package(Catch2 3 REQUIRED)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

target_link_libraries(tests PRIVATE pugixml::pugixml)

target_link_libraries(tests PRIVATE fmt::fmt)

target_link_libraries(tests PRIVATE unofficial::tinyexr::tinyexr)
