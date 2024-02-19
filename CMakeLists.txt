cmake_minimum_required(VERSION 3.27)
project(pt CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#[[Main executable]]

add_executable(pt
        src/main.cpp
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

        src/utils/basic_types.h
        src/utils/sampler.h
        src/utils/algs.h
        src/utils/chunk_allocator.h
        src/utils/render_threads.h

        src/io/scene_loader.cpp
        src/io/scene_loader.h
        src/io/image_writer.h
        src/io/progress_bar.h

        src/math/sampling.h
        src/math/vecmath.h
        src/math/math_utils.h
        src/math/transform.h
        src/math/piecewise_dist.cpp

        src/integrator/integrator.h
        src/integrator/utils.h
        src/integrator/light_sampler.cpp
        src/integrator/light_sampler.h
        src/integrator/integrator_type.h
        src/integrator/mis_nee_integrator.cpp
        src/integrator/intersection.h
        src/integrator/bdpt_nee_integrator.cpp

        src/geometry/geometry.h
        src/geometry/geometry.cpp

        src/scene/light.h

        src/color/rgb2spec.h
        src/color/sampled_spectrum.h
        src/color/cie_spectrums.h
        src/color/spectrum_consts.h
        src/color/color_space.h
        src/color/spectrum.cpp
        src/color/spectral_data.h
        src/color/spectrum.h

        src/materials/material.h
        src/materials/plastic.h
        src/materials/common.h
        src/materials/diffuse.h
        src/materials/dielectric.h
        src/materials/conductor.h
        src/materials/rough_conductor.h
        src/materials/trowbridge_reitz_ggx.h
        src/materials/rough_plastic.h
)

target_compile_options(pt PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)

target_link_options(pt PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
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

find_package(embree 4 REQUIRED)
target_link_libraries(pt PRIVATE embree)

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
        src/render_context.h
        src/framebuffer.h
        src/camera.h

        src/scene/emitter.h
        src/scene/envmap.h
        src/scene/texture.h
        src/scene/texture.cpp
        src/scene/scene.cpp
        src/scene/scene.h

        src/utils/basic_types.h
        src/utils/sampler.h
        src/utils/algs.h
        src/utils/chunk_allocator.h

        src/io/scene_loader.cpp
        src/io/scene_loader.h
        src/io/image_writer.h
        src/io/progress_bar.h

        src/math/sampling.h
        src/math/vecmath.h
        src/math/math_utils.h
        src/math/transform.h
        src/math/piecewise_dist.cpp

        src/integrator/integrator.h
        src/integrator/utils.h
        src/integrator/light_sampler.cpp
        src/integrator/light_sampler.h
        src/integrator/integrator_type.h
        src/integrator/intersection.h

        src/geometry/geometry.h
        src/geometry/geometry.cpp

        src/scene/light.h

        src/color/rgb2spec.h
        src/color/sampled_spectrum.h
        src/color/cie_spectrums.h
        src/color/spectrum_consts.h
        src/color/color_space.h
        src/color/spectrum.cpp
        src/color/spectral_data.h
        src/color/spectrum.h

        src/materials/material.h
        src/materials/plastic.h
        src/materials/common.h
        src/materials/diffuse.h
        src/materials/dielectric.h
        src/materials/conductor.h
        src/materials/rough_conductor.h
        src/materials/trowbridge_reitz_ggx.h
        src/materials/rough_plastic.h
        src/materials/test_ggx.cpp
        src/utils/tests.cpp
)

find_package(Catch2 3 REQUIRED)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

target_link_libraries(tests PRIVATE pugixml::pugixml)

target_link_libraries(tests PRIVATE fmt::fmt)

target_link_libraries(tests PRIVATE unofficial::tinyexr::tinyexr)
