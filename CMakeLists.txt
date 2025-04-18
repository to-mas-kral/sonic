cmake_minimum_required(VERSION 3.27)
project(pt CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")

#[[Main executable]]

add_library(pt_lib
        src/integrator_context.h
        src/framebuffer.h
        src/camera.h
        src/embree_accel.h
        src/settings.h
        src/renderer.h

        src/scene/emitter.h
        src/scene/envmap.h
        src/scene/texture.h
        src/scene/scene.cpp
        src/scene/scene.h
        src/scene/light.h
        src/scene/emitter.cpp
        src/scene/envmap.cpp
        src/scene/image.h
        src/scene/image.cpp
        src/scene/texture_types.h
        src/scene/texture.cpp
        src/scene/sphere_square_mapping.cpp
        src/scene/sphere_square_mapping.h
        src/scene/scene_attribs.h

        src/utils/basic_types.h
        src/utils/algs.h
        src/utils/thread_pool.h
        src/utils/thread_pool.cpp
        src/utils/panic.h
        src/utils/hasher.h
        src/utils/make_array.h
        src/utils/colormaps.h
        src/utils/float_comparison.h

        src/io/image_writer.h
        src/io/progress_bar.h

        src/math/sampling.h
        src/math/vecmath.h
        src/math/math_utils.h
        src/math/transform.h
        src/math/piecewise_dist.cpp
        src/math/sampling.cpp
        src/math/transform.cpp
        src/math/aabb.h
        src/math/discrete_dist.h
        src/math/coordinate_system.h
        src/math/samplers/sampler.cpp
        src/math/samplers/sampler.h
        src/math/samplers/halton_sampler.cpp
        src/math/samplers/halton_sampler.h
        src/math/samplers/sobol_matrices.h
        src/math/samplers/sobol_sampler.h

        src/integrator/integrator.h
        src/integrator/utils.h
        src/integrator/light_sampler.cpp
        src/integrator/light_sampler.h
        src/integrator/integrator_type.h
        src/integrator/mis_nee_integrator.cpp
        src/integrator/intersection.h
        src/integrator/shading_frame.h
        src/integrator/path_guiding_integrator.cpp
        src/integrator/path_guiding_integrator.h
        src/integrator/mis_nee_integrator.h
        src/integrator/integrator.cpp
        src/integrator/lambda_guiding_integrator.cpp
        src/integrator/lambda_guiding_integrator.h

        src/geometry/geometry_container.h
        src/geometry/geometry_container.cpp
        src/geometry/instance_id.h
        src/geometry/geometry_storage.cpp
        src/geometry/geometry_storage.h

        src/spectrum/rgb2spec.h
        src/spectrum/rgb2spec.cpp
        src/spectrum/spectral_quantity.h
        src/spectrum/spectral_quantity.cpp
        src/spectrum/cie_spectra.h
        src/spectrum/spectrum_consts.h
        src/spectrum/color_space.h
        src/spectrum/spectrum.cpp
        src/spectrum/spectral_data.h
        src/spectrum/spectrum.h
        src/spectrum/sampled_lambdas.cpp
        src/spectrum/sampled_lambdas.h

        src/materials/material.h
        src/materials/material.cpp
        src/materials/coateddiffuse.h
        src/materials/coateddiffuse.cpp
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
        src/materials/rough_coateddiffuse.h
        src/materials/rough_coateddiffuse.cpp
        src/materials/diffuse_transmission.cpp
        src/materials/diffuse_transmission.h
        src/materials/material_id.h

        src/pbrt_loader/pbrt_loader.cpp
        src/pbrt_loader/pbrt_loader.h
        src/pbrt_loader/lexer.cpp
        src/pbrt_loader/lexer.h
        src/pbrt_loader/stack_file_stream.h
        src/pbrt_loader/param.h

        src/path_guiding/sd_tree.cpp
        src/path_guiding/sd_tree.h
        src/path_guiding/binary_tree.cpp
        src/path_guiding/binary_tree.h
        src/path_guiding/sc_tree.cpp
        src/path_guiding/sc_tree.h

        src/gui/gui.cpp
        src/gui/gui.h
        src/gui/gui_state.h
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

find_package(spdlog CONFIG REQUIRED)
target_link_libraries(pt_lib PUBLIC spdlog::spdlog)

find_package(unofficial-miniply CONFIG REQUIRED)
target_link_libraries(pt_lib PUBLIC unofficial::miniply::miniply)

find_package(xxHash CONFIG REQUIRED)
target_link_libraries(pt_lib PRIVATE xxHash::xxhash)

find_package(embree 4 REQUIRED)
target_link_libraries(pt_lib PUBLIC embree)

find_package(OpenImageDenoise 2 REQUIRED)
target_link_libraries(pt_lib PUBLIC OpenImageDenoise)

find_package(SDL2 CONFIG REQUIRED)
target_link_libraries(pt_lib
        PRIVATE
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)

find_package(imgui REQUIRED)
target_link_libraries(pt_lib PRIVATE imgui::imgui)

find_package(implot REQUIRED)
target_link_libraries(pt_lib PRIVATE implot::implot)

find_package(OpenGL REQUIRED)
target_link_libraries(pt_lib PRIVATE OpenGL)

add_executable(pt
        src/main.cpp
)

configure_file(src/gui/Roboto-Medium.ttf font.ttf COPYONLY)

#[[RGB2SPEC]]
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

add_dependencies(pt_lib run_rgb2spec_opt)

target_link_libraries(pt PRIVATE pt_lib)

#[[Tests]]
add_executable(tests
        src/test/catch2_test_setup.cpp
        src/test/test_globals.h
        src/pbrt_loader/test_lexer.cpp
        src/pbrt_loader/test_params_parser.cpp
        src/pbrt_loader/test_pbrt_loader.cpp
        src/scene/test_sphere_square_mapping.cpp
        src/scene/test_envmap.cpp
        src/math/test_sampling_piecewise.cpp
        src/math/test_coordinate_system.cpp
        src/math/test_transform.cpp
        src/math/samplers/test_sobol_sampler.cpp
        src/utils/tests.cpp
        src/path_guiding/test_sd_tree.cpp
        src/path_guiding/test_binary_tree.cpp
        src/spectrum/test_spectrum.cpp
)

target_link_libraries(tests PRIVATE pt_lib)

find_package(Catch2 3 REQUIRED)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

target_link_libraries(tests PRIVATE fmt::fmt)

target_link_libraries(tests PRIVATE unofficial::tinyexr::tinyexr)
