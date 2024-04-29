#ifndef SCENE_ATTRIBS_H
#define SCENE_ATTRIBS_H

#include <filesystem>

#include "../math/transform.h"
#include "../utils/basic_types.h"

struct CameraAttribs {
    static CameraAttribs
    pbrt_defaults() {
        return CameraAttribs{
            .fov = 90.0f,
            .camera_to_world = mat4::identity(),
        };
    }

    f32 fov = 30.f;
    mat4 camera_to_world = mat4::identity();
};

struct FilmAttribs {
    static FilmAttribs
    pbrt_defaults() {
        return FilmAttribs{
            .resx = 1280,
            .resy = 720,
        };
    }

    u32 resx = 1280;
    u32 resy = 720;
    std::filesystem::path filename{};
};

struct SceneAttribs {
    static SceneAttribs
    pbrt_defaults() {
        return SceneAttribs{
            .camera = CameraAttribs::pbrt_defaults(),
            .film = FilmAttribs::pbrt_defaults(),
            .max_depth = 5,
        };
    }

    CameraAttribs camera{};
    FilmAttribs film{};
    u32 max_depth = 0;
};

#endif // SCENE_ATTRIBS_H
